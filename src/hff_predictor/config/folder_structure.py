import os
import logging

LOGGER = logging.getLogger(__name__)


def create_folder_structure(store_location: str = "U:"):
    if store_location is "find":
        cwd = os.getcwd()
        LOGGER.debug("Voorspelmodel staat nu geplaatst in :{}".format(cwd))

        root_disk = cwd.split(":")[0] + ":"
        LOGGER.debug("De schijf waar alles wordt opgeslagen is: {}".format(root_disk))

    else:
        root_disk = store_location
        LOGGER.debug("De schijf waar alles wordt opgeslagen is: {}".format(root_disk))

    def create_folder(dir_path: str):
        try:
            os.makedirs(dir_path)

        except FileExistsError:
            LOGGER.debug("{} bestaal al".format(dir_path))

    main_folder = "Productie Voorspelmodel"
    project_folder = root_disk + "\\" + main_folder
    create_folder(dir_path=project_folder)

    # De 3 subfolders
    input_folder = "Input"
    processed_folder = "Processed"
    output_folder = "Output"

    input_subfolders = ["Bestellingen", "Campagnes", "Productstatus"]
    processed_subfolders = ["Bestellingen", "Campagnes", "Features", "Superunie", "Weer"]
    output_subfolders = ["Testvoorspellingen", "Tussenresultaten", "Voorspellingen"]

    bestellingen_subfolders = ["Actief", "Consumentgroepnummer", "Inactief", "Standaard", "Superunie eigenschappen"]
    voorspellingen_subfolders = ["Performance"]

    # Input folders
    for i in input_subfolders:
        dir_name = project_folder + "\\" + input_folder + "\\" + i
        create_folder(dir_path=dir_name)

    for i in processed_subfolders:
        dir_name = project_folder + "\\" + processed_folder + "\\" + i
        create_folder(dir_path=dir_name)

    for i in output_subfolders:
        dir_name = project_folder + "\\" + output_folder + "\\" + i
        create_folder(dir_path=dir_name)

    # Sub sub folders
    for j in bestellingen_subfolders:
        dir_name = project_folder + "\\" + processed_folder + "\\" + "Bestellingen" + "\\" + j
        create_folder(dir_path=dir_name)

    for j in voorspellingen_subfolders:
        dir_name = project_folder + "\\" + output_folder + "\\" + "Tussenresultaten" + "\\" + j
        create_folder(dir_path=dir_name)


def init_folder_setup(output):

    create_folder_structure(store_location=output)
