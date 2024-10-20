from datetime import datetime
import json


class QAEvaluation:
    def __init__(self, args, datasets, model ) -> None:
        self.args = args
        self.datasets = datasets
        self.model = model
        # 定义日志信息
        self.log_info = """
        Evaluation Log
        --------------
        
        Model Information:
        - Model Name: {model}
        Data Information:
        - Dataset Name: {datasets}
        - Dataset Num: {nums}
        
        Evaluation Results:
        - Evaluation Date: {time}
        - Evaluation Metrics:
        - Accuracy: {accuracy}
        """
    def run(self, run_results_path) -> float:
        
        with open(run_results_path, 'r') as file:
            instances = json.load(file)

        correct_count = 0
        logs_result = []
        for idx, inst in enumerate(instances):
            response_correct = inst['extract_answer']==inst['answer']
            logs_result.append(inst.to_dict())
            if response_correct:
                correct_count += 1
        
        accuracy = correct_count / len(instances) 
        

        log_info = self.log_info.format(model = self.args.model,
                                        datasets = self.args.dataset,
                                        nums = len(instances),
                                        time = datetime.now(),
                                        accuracy = accuracy)
        
        with open(self.args.output_dir+'logs.txt', 'a', encoding='utf-8') as file:
            file.write(log_info)
    






















