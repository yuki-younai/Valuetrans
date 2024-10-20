from datetime import datetime
import json

class  MutiChoiceEvaluation:
    def __init__(self, args, datasets, model, help_model ) -> None:
        self.args = args
        self.datasets = datasets
        self.model = model
        self.help_model = help_model

    def run(self, run_results_path) -> float:

        with open(run_results_path, 'r') as file:
            instances = json.load(file)

        if self.datasets.name in ["mfq-30", "pvq-rr"]:
            param_dict = self.datasets.evaluation(instances)
            temp_dict = {"role": instances[0]['role'],
                         "model":self.args.model,
                         "datasets":self.args.dataset,
                         "nums":len(instances),
                         "time":datetime.now()}
            param_dict.update(temp_dict)
            logs_info = self.datasets.logs_info.format(**param_dict)

        # 写入文件
        with open(self.args.output_dir+'logs.txt', 'a', encoding='utf-8') as file:
            file.write(logs_info)


















