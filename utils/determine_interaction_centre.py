import re
import subprocess
from pathlib import Path

import Bio.PDB
import numpy as np
import scipy.spatial


class ChainSelect(Bio.PDB.Select):
    """Class to select chains in Bio.PDB object and save them to a new pdb file

    Args:
        chain_ls (list): List of the chain IDs in the pdb file
    """

    def __init__(self, chain_ls: list):
        self.chain_ls = chain_ls

    def accept_chain(self, chain):
        if chain.get_id() in self.chain_ls:
            return 1
        else:
            return 0


def split_pdb(pdb_filepath: Path, pair_chains: dict):
    """Split a pdb file by seperating the binding partners into seperate files.

    Args:
        pdb_filepath (Path): Input pdb file.
        pair_chains (dict): dict of binding_partner/chain_ids pairs.

    Returns:
        dict: dict of binding_partner/filename pairs
    """
    # read in the pdb chains individually
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("in_struc", pdb_filepath)

    io = Bio.PDB.PDBIO()
    io.set_structure(structure)

    outname_dict = {}
    for bp, chains in pair_chains.items():
        chain_select = ChainSelect(chains)
        outname = str(pdb_filepath) + f".{bp}"
        io.save(outname, select=chain_select)
        outname_dict[bp] = outname

    return outname_dict


def run_parse_psa(pdb_filepath: Path):
    """Wrapper to run and parse PSA on a pdb file.

    Args:
        pdb_filepath (Path): Path to the input pdb file.

    Returns:
        list: Parsed PSA output.
    """
    psa_out = run_psa(pdb_filepath)
    parsed_psa_out = parse_psa(psa_out, pdb_filepath)

    return parsed_psa_out


def run_psa(pdb_filepath: Path):
    """Run the PSA executable on pdb_filepath.

    Args:
        pdb_filepath (Path):  Path to the input pdb file.
    """

    psa_out = str(pdb_filepath) + ".psa.out"
    psa_err = str(pdb_filepath) + ".psa.err"

    with open(psa_out, "w") as psa_out_handle, open(psa_err, "w") as psa_err_handle:
        arg_ls = ["psa", "-t", "-nh", "-w", "-v", str(pdb_filepath)]
        retcode = subprocess.call(arg_ls, stdout=psa_out_handle, stderr=psa_err_handle)
        if retcode != 0:
            raise ValueError("Error while running psa, check error file.")

    return psa_out


def parse_psa(psa_outfile: str, pdb_file: str):
    """Parse the output from the PSA algorithm, identifying

    Args:
        psa_outfile (str): Path to the psa output.
        pdb_file (str): Path to the pdb file the psa was run on.

    Returns:
        list: list of lists of exposed residues per chain
    """
    # check where the pdb switches chain
    with open(pdb_file) as inf:
        new_chain_starts = []
        lines = inf.readlines()
        for ind, line in enumerate(lines):
            if (
                line.startswith("ATOM")
                and ind > 0
                and (
                    lines[ind - 1].startswith("ATOM")
                    or lines[ind - 1].startswith("TER")
                )
            ):
                chain = line[21]
                if chain != lines[ind - 1][21]:
                    switch_num = int(re.sub("[^0-9]", "", line[22:26].strip()))
                    previous_num = int(
                        re.sub("[^0-9]", "", lines[ind - 1][22:26].strip())
                    )
                    switch_res = line[17:20]

                    # only rely on this if the switch would not be registered by non-increasing
                    # criteria (otherwise potential for bug when both start with the same aa)
                    if previous_num < switch_num:
                        new_chain_starts.append((switch_num, switch_res))

    # parse psa output
    chain_lists = []
    res_list = []
    line_ind = 0
    previous_line = ""
    with open(psa_outfile) as psa_in:
        psa_lines = psa_in.readlines()
        for line in psa_lines:
            if line.startswith("ACCESS"):

                line_res_num = int(line[6:11].strip())
                line_res_three_letter = line[14:17]
                line_tuple = (line_res_num, line_res_three_letter)

                if line_ind > 0:
                    if (line_tuple in new_chain_starts) or (
                        int(previous_line[6:11].strip()) > line_res_num
                    ):
                        chain_lists.append(res_list)
                        res_list = []

                # if > 7.5% relative accessible surface area --> residue exposed
                acc_per = float(line[61:67])
                if acc_per > 7.5:
                    res_number = line[6:12].strip()
                    res_list.append(res_number)
                line_ind += 1
                previous_line = line

    if len(res_list) > 0:
        chain_lists.append(res_list)

    check_sum = 0
    for chain_res_list in chain_lists:
        check_sum += len(chain_res_list)
        if check_sum == 0:
            raise AssertionError("PSA did not produce output")

    return chain_lists


def get_centre_coordinates(
    split_pdb_dict: dict, parsed_psa_output: dict, distance_cutoff: int = 4
):
    """Determine the coordinates of the interaction centre between two protein binding partners.

    Args:
        split_pdb_dict (dict): Dict returned by split_pdb()
        parsed_psa_output (dict): Parsed psa output returned by run_parse_psa()
        distance_cutoff (int, optional): Max distance for two atoms to be considered interacting.
            Defaults to 4.

    Raises:
        ValueError: The two binding partners do not have contacts at distance_cutoff.

    Returns:
        np.array: [x,y,z] - the interaction centre
    """
    coord_dict = {}
    for bp in split_pdb_dict.keys():
        coord_list = []
        psa_chain_list = parsed_psa_output[bp]

        chain_count = 0
        res_list = psa_chain_list[chain_count]
        current_chain = None
        with open(split_pdb_dict[bp]) as pdb_in:
            for line in pdb_in.readlines():
                if line.startswith("ATOM"):

                    # check which chain we are on and choose the right residue list
                    if current_chain is None:
                        current_chain = line[21]
                    else:
                        if current_chain != line[21]:
                            current_chain = line[21]
                            chain_count += 1
                            try:
                                res_list = psa_chain_list[chain_count]
                            except IndexError:
                                # print(split_pdb_dict)
                                # sys.stdout.flush()
                                raise IndexError("IndexError occured")

                    # check if the current residue is surface exposed, if so add atom coordinates
                    res_number = line[22:27].strip()
                    if res_number in res_list:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coord_list.append([x, y, z])
        coord_dict[bp] = np.array(coord_list)

    # calculate distances
    dist_mat = scipy.spatial.distance.cdist(coord_dict["bp1"], coord_dict["bp2"])
    bp1_dist = np.min(dist_mat, axis=1)
    bp2_dist = np.min(dist_mat, axis=0)

    interacting_atom_coords_bp1 = coord_dict["bp1"][bp1_dist <= distance_cutoff, :]
    interacting_atom_coords_bp2 = coord_dict["bp2"][bp2_dist <= distance_cutoff, :]

    interaction_center_bp1 = np.mean(interacting_atom_coords_bp1, axis=0)
    interaction_center_bp2 = np.mean(interacting_atom_coords_bp2, axis=0)

    if len(interacting_atom_coords_bp1) == 0:
        raise ValueError("No contacts at current distance cutoff.")

    center_final = (interaction_center_bp1 + interaction_center_bp2) / 2

    return center_final
