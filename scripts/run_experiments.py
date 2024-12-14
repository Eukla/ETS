import os


methods = {
    "ects": "ects -u 0.0",
    "edsccplus": "--cplus edsccplus",
    "ecec": "--java ecec",
    "teaser": "--java teaser -s 20",
    "mlstm": "mlstm",
    "economy-k": "economy-k",
    "calimera": "calimera -dp 1.0"
}

DATA_PATH = './data'
RESULTS_PATH = './results'

datasets = {
    "BasicMotions": f"{DATA_PATH}/UCR_UEA/BasicMotions/BasicMotions",
    "DodgerLoopDay": f"{DATA_PATH}/UCR_UEA/DodgerLoopDay/DodgerLoopDay",
    "DodgerLoopGame": f"{DATA_PATH}/UCR_UEA/DodgerLoopGame/DodgerLoopGame",
    "DodgerLoopWeekend": f"{DATA_PATH}/UCR_UEA/DodgerLoopWeekend/DodgerLoopWeekend",
    "HouseTwenty": f"{DATA_PATH}/UCR_UEA/HouseTwenty/HouseTwenty",
    "LSST": f"{DATA_PATH}/UCR_UEA/LSST/LSST",
    "PickupGestureWiimoteZ": f"{DATA_PATH}/UCR_UEA/PickupGestureWiimoteZ/PickupGestureWiimoteZ",
    "PLAID": f"{DATA_PATH}/UCR_UEA/PLAID/PLAID",
    "PowerCons": f"{DATA_PATH}/UCR_UEA/PowerCons/PowerCons",
    "SharePriceIncrease": f"{DATA_PATH}/UCR_UEA/SharePriceIncrease/SharePriceIncrease",
    "Maritime": f"{DATA_PATH}/maritime_30_CA.csv",
    "BioArchive": f"{DATA_PATH}/BioArchive/MTS.csv",
}


if __name__ == '__main__':
    configurations = []
    for method_name, method_args in methods.items():
        dash_g_argument = 'normal' if method_name=='mlstm' else 'vote'
        for dataset_name, dataset_path in datasets.items():
            if dataset_name == 'Maritime':
                dataset_args = f"-i {dataset_path} -g {dash_g_argument} -v 5 -d 0 -c -1"
            elif dataset_name == 'BioArchive':
                dataset_args = f"-i {dataset_path} -g {dash_g_argument} -v 3 -d 0 -c -1"
            else:
                dataset_args = f"-t {dataset_path}_TRAIN.arff -e {dataset_path}_TEST.arff \
                    -g {dash_g_argument} --make-cv -h Class -c -1"
            full_args = f"{dataset_args} {method_args}"
            configurations.append((method_name, dataset_name, full_args))

    print(f"Starting the launch of {len(configurations)} configurations")
    print(f"({len(methods)} methods x {len(datasets)} datasets)")
    print(f"Logging results to {RESULTS_PATH}")
    for method_name, dataset_name, full_args in configurations:
        output_file_path = f"{RESULTS_PATH}/{method_name}_{dataset_name}.log"
        print(f"Running ets -o {output_file_path} {full_args}")
        os.system(f"ets -o {output_file_path} {full_args}")
