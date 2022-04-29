# -*- coding: utf-8 -*-
"""DOWNLOAD ROBUST-MIS CHALLENGE DATA

1. Create an account for synapse
2. Register for the Challenge: https://www.synapse.org/#!Synapse:syn18779624/wiki/591266
3. Install the synapse client: `pip install synapseclient`
4. Insert the files_synapse_id as project_id below
5.  Add your credentials at the end of the script
6. Run the script and get the data
7. Have fun :-)

"""

import synapseclient
import synapseutils


def download_data(email, password, local_folder):
    print("Start downloading")

    # login to Synapse
    syn = synapseclient.login(email=email, password=password, rememberMe=True)

    # download all the files in folder files_synapse_id to a local folder
    project_id = "syn21891314"  # this is the project id of the files.
    all_files = synapseutils.syncFromSynapse(syn,
                                             entity=project_id,
                                             path=local_folder)

    print("Finished downloading")


if __name__ == "main":

    # settings
    local_folder = "/mnt/disks/rmis_test"
    email = "saicharanb56@gmail.com"
    password = "robustMIS2022"

    # download data
    download_data(email, password, local_folder)
