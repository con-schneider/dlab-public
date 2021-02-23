"""
This pipeline takes a set of docking pairings of antigen and antibody files, prepares them for
docking depending on the constraints given, then docks them and parses them for input into the
DLAB pipeline and saves the docking scores.
"""
import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import openbabel
import pandas as pd
import yaml
from biopandas.pdb import PandasPdb
from joblib import delayed
from joblib import Parallel

from utils.determine_interaction_centre import get_centre_coordinates
from utils.determine_interaction_centre import run_parse_psa
from utils.determine_interaction_centre import split_pdb
from utils.typer import Typer


def get_args():
    """Parse command line args

    Returns:
        argparse.Namespace: parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", help="Config yaml file", required=True)
    args = parser.parse_args()

    return args


def parse_yaml(yaml_path: Path):
    """Parses yaml config file

    Args:
        yaml_path (Path): Path to the yaml file

    Returns:
        dict: parsed config
    """

    with open(yaml_path) as yaml_file:
        parsed_yaml = yaml.safe_load(yaml_file)

    return parsed_yaml


def check_adjust_pair_chains(pair: tuple, output_dir: Path):
    """Ensure that the chain IDs in the two binding partners are non-overlapping. Creates an
        alternatively labelled pdb file in case they do.

    Args:
        pair (tuple): Tuple of paths to the pdb files for the two binding partners.

    Returns:
        tuple: pair, pair_chains, where pair is the paths to the (adjusted) pdb files and
            pair_chains are the (adjusted) chain IDs.
    """

    bp1_path = pair[0]
    bp2_path = pair[1]

    bp1_frame = PandasPdb().read_pdb(bp1_path).df["ATOM"]
    bp2_frame = PandasPdb().read_pdb(bp2_path).df["ATOM"]

    bp1_chains = bp1_frame.chain_id.unique()
    bp2_chains = bp2_frame.chain_id.unique()
    pair_chains = {"bp1": list(bp1_chains), "bp2": list(bp2_chains)}

    # if there is overlap, adjust bp2
    adjust = False
    for chain in bp2_chains:
        if chain in bp1_chains:
            adjust = True
            break

    if adjust:
        new_bp2_chains = {}
        bp1_chains_ord = [ord(char) for char in bp1_chains]
        current_num = ord("A")
        # choose new chain IDs for bp2
        for chain in bp2_chains:
            while current_num in bp1_chains_ord:
                current_num += 1
            new_chain = chr(current_num)
            current_num += 1
            new_bp2_chains[chain] = new_chain

        # write an alternative pdb file to dock with, using the new chain IDs
        with open(bp2_path) as inf:
            lines = inf.readlines()
        with open(
            output_dir / "alt_pdbs_for_docking" / Path(bp2_path).name, "w"
        ) as outf:
            for line in lines:
                if line.startswith("ATOM") or line.startswith("TER"):
                    chain = line[21]
                    line = list(line)
                    line[21] = new_bp2_chains[chain]
                    line = "".join(line)
                outf.write(line)

        # add the new adjusted file to the pair
        pair = (pair[0], output_dir / "alt_pdbs_for_docking" / Path(bp2_path).name)
        pair_chains = {
            "bp1": list(bp1_chains),
            "bp2": [new_bp2_chains[chain] for chain in bp2_chains],
        }

    return pair, pair_chains


def run_docking(pair: tuple, config: dict, output_dir: Path, timeout: int = -1):
    """Run docking algorithm on one pair of binding partners.

    Args:
        pair (tuple): Tuple of paths to the pdb files for the two binding partners.
        config (dict): Parsed yaml config
        output_dir (Path): Output directory
        timeout (int, optional): Terminate the docking after n seconds. If -1, no timeout is
            applied. Defaults to -1.

    Returns:
        dict: dictionary with keys = [ret_code, pair_paths, pair_chains, output_name]
                ret_code: return code, 0 if sucessful, 1 otherwise
                pair: paths to the (adjusted) pdb files
                pair_chains: (adjusted) chain IDs
                output_name: path to docking output file
    """
    # check if output exists and that there is no clash regarding the chain allocation
    bp1 = Path(pair[0])
    bp2 = Path(pair[1])

    pair, pair_chains = check_adjust_pair_chains(pair, output_dir)

    output_name = f"{bp1.name.split('.')[0]}_{bp2.name.split('.')[0]}.out"
    output_name = output_dir / "docking_output" / output_name
    # if docking has already been done, we can skip this
    if not config["ignore_repeats"]:
        if output_name.exists():
            print(f"{output_name} exists.")
            sys.stdout.flush()
            return {
                "return_code": 0,
                "pair": pair,
                "pair_chains": pair_chains,
                "output_name": output_name,
            }

    bp1 = pair[0]  # the ab
    bp2 = pair[1]  # the ag

    # this is necesary because zdock internally limits the size of the array that holds the command
    # line arguments --> if the output name is too long, it becomes garbled
    output_name_temp = "./" + output_name.name

    proc = subprocess.Popen(
        [
            "./external/zdock-3.0.2-src/zdock",
            "-R",
            bp2,
            "-L",
            bp1,
            "-o",
            output_name_temp,
        ]
    )
    if timeout > 0:
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("Killed due to timeout")
            return {
                "return_code": 1,
                "pair": pair,
                "pair_chains": pair_chains,
                "output_name": output_name,
            }
    else:
        proc.communicate()

    shutil.move(str(output_name_temp), str(output_name))
    return {
        "return_code": 0,
        "pair": pair,
        "pair_chains": pair_chains,
        "output_name": output_name,
    }


def make_pdbs(docking_file: Path, config: dict, output_dir: Path):
    """Generate pdb files from docking output.

    Args:
        docking_file (Path): Path to ZDock output file.
        config (dict): Parsed yaml config.

    Returns:
        Path: the path to the directory in which the pdb files are located.
    """
    # create the shortened outfile --> only generating the top num_poses
    with open(docking_file) as inf:
        lines = inf.readlines()
    shortened_file = str(docking_file) + ".short"
    with open(shortened_file, "w") as outf:
        for line in lines[: 5 + config["num_poses"]]:
            outf.write(line)

    cwd = Path(os.getcwd())

    # set up the temporary directory in which to create the pdbs
    tmp_dir = output_dir / "pdbs" / docking_file.name.split(".")[0]
    tmp_dir.mkdir(exist_ok=True)

    # copy over relevant ZDock helper code and run
    shutil.copy(cwd / "external/zdock-3.0.2-src/create_lig", tmp_dir / "create_lig")
    shutil.copy(cwd / "external/zdock-3.0.2-src/create.pl", tmp_dir / "create.pl")
    subprocess.call(
        f"chmod +x {tmp_dir / 'create.pl'} {tmp_dir / 'create_lig'}", shell=True
    )

    os.chdir(tmp_dir)

    subprocess.call(f"./create.pl {shortened_file}", shell=True)

    os.chdir(cwd)

    return tmp_dir


def teardown(docking_output: dict, output_dir: Path):
    """Delete pdb files and psa output in the output_dir.

    Args:
        docking_output (dict): Output from run_docking().
        output_dir (Path): The directory in which pdb files etc have been created.
    """

    docking_output_filepath = docking_output["output_name"]
    tmp_dir = output_dir / "pdbs" / docking_output_filepath.name.split(".")[0]
    shutil.rmtree(tmp_dir)
    return 0


def generate_types_files(docking_output: dict, config: dict, output_dir: Path):
    """Generates pdbs from docking output file, converts to more storage efficient gninatypes files
        and deletes the pdbs.

    Args:
        docking_output (dict): Output from run_docking().
        config (dict): Parsed yaml config.
        output_dir (Path): Directory in which to generate the gninatypes files.
    """

    docking_output_filepath = docking_output["output_name"]
    pair_chains = docking_output["pair_chains"]

    pair_pdb_dir = make_pdbs(docking_output_filepath, config, output_dir)

    complex_pdbs = pair_pdb_dir.glob("*.pdb")

    parsed_psa_output = None
    for complex_pdb in complex_pdbs:
        # split pdb
        split_pdb_dict = split_pdb(complex_pdb, pair_chains)

        # get surface exposed atoms on both interaction partners (only need to run this once)
        if parsed_psa_output is None:
            parsed_psa_output = {}
            for bp, split_pdb_path in split_pdb_dict.items():
                try:
                    parsed_psa_output[bp] = run_parse_psa(split_pdb_path)
                except AssertionError:  # PSA returned empty file
                    with open(output_dir / "log" / "errors.csv", "a") as err_out:
                        err_out.write(f"{split_pdb_path},PSA emtpy file\n")
                    return 1
                except ValueError:  # PSA returned 0
                    with open(output_dir / "log" / "errors.csv", "a") as err_out:
                        err_out.write(f"{split_pdb_path},PSA error\n")
                    return 2

        # determine center and parse to types format
        try:
            centre_coords = get_centre_coordinates(split_pdb_dict, parsed_psa_output)
        except ValueError:  # no interaction at cutoff
            with open(output_dir / "log" / "errors.csv", "a") as err_out:
                err_out.write(f"{split_pdb_dict['bp1']},no interaction at cutoff\n")
            return 3

        typer = Typer()
        for bp, split_pdb_file in split_pdb_dict.items():
            # antibody is bp1, antigen is bp2
            if bp == "bp1":
                add = "ab"
            else:
                add = "ag"
            name_root = docking_output_filepath.name.split(".")[0]

            outdir_types = output_dir / "gninatypes" / name_root
            outdir_types.mkdir(exist_ok=True)
            outpath = outdir_types / Path(split_pdb_file).name.replace(
                f".pdb.{bp}", f"_{add}.gninatypes"
            )

            # supress warnings from openbabel due to pdbs formatted for docking
            openbabel.obErrorLog.SetOutputLevel(openbabel.obError)
            typer.run(split_pdb_file, outpath, centre_coords)

    # delete files
    teardown(docking_output, output_dir)
    return 0


def run(config: dict, output_dir: Path):
    """Run the pipeline as specified by the config.

    Args:
        config (dict): Parsed yaml config.
        output_dir (Path): Directory in which to create all run files.
    """

    # for each input file, run dock and collect the docking output
    docking_pairs = pd.read_csv(config["pairings_file"])
    docking_pairs = [(row[1]["bp1"], row[1]["bp2"]) for row in docking_pairs.iterrows()]

    with Parallel(n_jobs=config["n_jobs"], verbose=1) as parallel:
        timeout = config["docking_timeout"]
        docking_returns = parallel(
            delayed(run_docking)(pair, config, output_dir, timeout)
            for pair in docking_pairs
        )

    # check if any docking attempts timed out
    timeouts_pairs = []
    finished_docks_output = []
    for ret in docking_returns:
        if ret["return_code"] == 1:
            timeouts_pairs.append(ret["pair"])
        else:
            finished_docks_output.append(ret)

    if config["repeat_timeout_docks"]:
        # repeat timed out docks without timeout imposed
        while len(timeouts_pairs) > 0:
            print(f"Rerunning {len(timeouts_pairs)} docks that timed out.")
            with Parallel(n_jobs=config["n_jobs"], verbose=1) as parallel:
                docking_returns = parallel(
                    delayed(run_docking)(pair, config, output_dir)
                    for pair in timeouts_pairs
                )
            timeouts_pairs = []
            for ret in docking_returns:
                if ret["return_code"] == 1:
                    timeouts_pairs.append(ret["pair"])
                else:
                    finished_docks_output.append(ret)

    else:  # log the skipped docks
        with open(output_dir / "log" / "timed_out_docks.csv", "w") as outf:
            outf.write("bp1,bp2\n")
            for pair in timeouts_pairs:
                outf.write(f"{pair[0]},{pair[1]}")

    # generate pdbs, parse to gninatypes format
    with Parallel(n_jobs=config["n_jobs"], verbose=1) as parallel:
        parallel(
            delayed(generate_types_files)(docking_output, config, output_dir)
            for docking_output in finished_docks_output
        )

    # tar the gninatypes output, then delete
    os.chdir(output_dir)
    to_tar = "./gninatypes"
    tarred_name = f"./{config['run_name']}_gninatypes.tar.gz"

    proc = subprocess.Popen(["tar", "-czf", tarred_name, to_tar])
    proc.communicate()
    shutil.rmtree(to_tar)
    return 0


def setup_run(config: defaultdict, args: argparse.Namespace):
    """Setup the run folders.

    Args:
        config (defaultdict): Parsed yaml config
        args (argparse.Namespace): command line arguments
    """
    now = datetime.now()
    output_dir = Path(config["output_directory"]) / f"{config['run_name']}"
    output_dir.mkdir(exist_ok=True, parents=True)

    log_path = output_dir / "log"
    log_path.mkdir(exist_ok=True, parents=True)

    dock_path = output_dir / "docking_output"
    dock_path.mkdir(exist_ok=True, parents=True)

    pdb_path = output_dir / "pdbs"
    pdb_path.mkdir(exist_ok=True, parents=True)

    types_path = output_dir / "gninatypes"
    types_path.mkdir(exist_ok=True, parents=True)

    alt_path = output_dir / "alt_pdbs_for_docking"
    alt_path.mkdir(exist_ok=True, parents=True)

    # save and timestamp the config file used in the run
    now = datetime.now()
    shutil.copy(
        args.c,
        output_dir / "log" / f"config_{now.strftime(format='%Y-%m-%d_%H-%M-%S')}.yaml",
    )
    return output_dir


if __name__ == "__main__":
    args = get_args()
    config = parse_yaml(args.c)
    output_dir = setup_run(config, args)
    run(config, output_dir)
