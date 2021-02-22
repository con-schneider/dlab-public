import os


def test_psa():
    """Test presence of the psa executable"""
    root = os.getcwd()
    psa_path = root + "/external/psa_execs/psa"
    assert os.path.isfile(psa_path)


def test_zdock_present():
    """Test presence of all files required for docking"""
    root = os.getcwd()
    zdock_path = root + "/external/zdock-3.0.2-src/"
    for f in ["create_lig", "create.pl", "mark_sur", "zdock", "uniCHARMM"]:
        assert os.path.isfile(zdock_path + f)
