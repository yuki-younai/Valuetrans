import json
import argparse
from easyEval.utils.utils import init_output, set_seed
from easyEval.Dataset import str2datasets
from easyEval.Model import API_MODEL, str2models
from easyEval.Run import str2runs
from easyEval.Evaluation import str2evaluation

def main():
    
   parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   #1.基本选择：选择模型，数据集，模型配置，训练方式
   parser.add_argument("--dataset", type=str, default="gsm8k",
                     help="Path of the preprocessed dataset.")
   parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                     help="Path of the preprocessed dataset.")
   parser.add_argument("--method", type=str, default="base",
                     help="Path of the preprocessed dataset.")
   parser.add_argument("--run_results", type=str, default=None,
                     help="Path of the preprocessed dataset.")
   parser.add_argument("--ratio", type=float, default=0.1,
                    help="Random seed.")
   parser.add_argument("--gpu", type=int, default=3,
                    help="Random seed.")
   args = parser.parse_args()
   with open("easyEval/config.json", 'r') as file:
      args.config = json.load(file)
   
   ################
   #Init Environment
   ################
   set_seed(42)
   init_output(args)
   ################
   #Init Dataset And Model
   ################
   dataset = str2datasets[args.dataset]()
   if args.model in API_MODEL:
      model = str2models["api"](args.model, api_key=args.config['openai_api_key'], base_url=args.config['base_url'])
   else:
      model = str2models["local"](args)
   help_model = model = str2models["api"]("gpt-3.5-turbo", api_key=args.config['openai_api_key'], base_url=args.config['base_url'])
   if args.run_results is None:
      ################
      #Run Experiment
      ################  
      instances = dataset.init_instance(args.ratio)
      instances = str2runs[args.method](args, instances, dataset, model, help_model)
   ################
   #Evaluation
   ################  
   evaluation = str2evaluation[dataset.type](args, dataset, model, help_model )
   evaluation.run(args.run_results)


if __name__ == "__main__":
    main()




















