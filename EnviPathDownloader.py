"""
PathFormer
Copyright (C) 2026  Liam Brydon
Contact at: liam.brydon@auckland.ac.nz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from enviPath_python.enviPath import *
from enviPath_python.objects import *
import json
import os
from tqdm import tqdm


def download_envipath(save_path, force_download=False):
    host = 'https://envipath.org/'
    bbd_package = f'{host}package/32de3cf4-e3e6-4168-956e-32fa5ddb0ce1'
    soil_package = f'{host}package/5882df9c-dae1-4d80-a40e-db4724271456'
    sludge_package = f'{host}package/521c547a-fd2a-491c-ad5b-7eaa1577fb65'
    sediment_package = f'{host}package/f05e38d8-e9b4-4c3e-b0d8-9ab29966eccf'
    with open("ep_login") as l_file:
        file = l_file.readline().strip()
    username, password = file.split(" ")
    eP = enviPath('https://envipath.org/api/legacy', new_api=True)
    eP.login(username, password)
    print(eP.who_am_i())
    packages = [(bbd_package, "bbd"), (soil_package, "soil"), (sludge_package, "sludge"), (sediment_package, "sediment")]
    for package, name in packages:
        print(f"Starting {name} download.")
        local_package_path = os.path.join(save_path, f"{name}.json")
        if os.path.exists(local_package_path) and not force_download:
            print(f"Package {name} has already been downloaded.")
            continue
        envi_package = eP.get_package(package)
        compounds = envi_package.get_compounds()
        reactions = envi_package.get_reactions()
        rules = envi_package.get_rules()
        pathways = envi_package.get_pathways()
        scenarios = envi_package.get_scenarios()
        data_dict = {"compounds": [], "reactions": [], "rules": [], "pathways": [], "scenarios": []}
        for rule in tqdm(rules, desc="Getting rules"):
            if type(rule) is SimpleRule:
                try:
                    data_dict["rules"].append(rule.get_json())
                except Exception as e:
                    print(f"Failed to get {rule} with error {e}")
        for compound in tqdm(compounds, desc="Getting compounds"):
            try:
                data_dict["compounds"].append(compound.get_json())
            except Exception as e:
                print(f"Failed to get {compound} with error {e}")
        for reaction in tqdm(reactions, desc="Getting reactions"):
            try:
                data_dict["reactions"].append(reaction.get_json())
            except Exception as e:
                print(f"Failed to get {reaction} with error {e}")
        for pathway in tqdm(pathways, desc="Getting pathways"):
            try:
                data_dict["pathways"].append(pathway.get_json())
            except Exception as e:
                print(f"Failed to get {pathway} with error {e}")
        for scenario in tqdm(scenarios, desc="Getting scenarios"):
            try:
                data_dict["scenarios"].append(scenario.get_json())
            except Exception as e:
                print(f"Failed to get {scenario} with error {e}")
        with open(local_package_path, 'w') as data_file:
            json.dump(data_dict, data_file, indent=4)


def check_envipath_data():
    envipath_data_path = "data/envipath"
    if any(not os.path.exists(f"data/envipath/{package}.json") for package in ["soil", 'bbd', 'sludge', 'sediment']):
        os.makedirs(envipath_data_path, exist_ok=True)
        download_envipath(envipath_data_path)
    return


def download_raw():
    eP = enviPath('https://envipath.org')
    bbd = eP.get_package('https://envipath.org/package/32de3cf4-e3e6-4168-956e-32fa5ddb0ce1')
    soil = eP.get_package('https://envipath.org/package/5882df9c-dae1-4d80-a40e-db4724271456')
    sludge = eP.get_package('https://envipath.org/package/521c547a-fd2a-491c-ad5b-7eaa1577fb65')
    sediment = eP.get_package('https://envipath.org/package/f05e38d8-e9b4-4c3e-b0d8-9ab29966eccf')
    packages = [("BBD", bbd), ("SOIL", soil), ("SLUDGE", sludge), ("SEDIMENT", sediment)]
    for name, package in tqdm(packages):
        with open(f"data/envipath/EAWAG-{name}.json", "w") as j_file:
            json.dump(package.export_as_json(), j_file)


if __name__ == "__main__":
    check_envipath_data()
