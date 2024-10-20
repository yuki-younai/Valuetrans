from easyEval.Run.BaseRun import BaseRun
from easyEval.Run.VoteRun import VoteRun

str2runs = { "base": BaseRun, "vote":VoteRun}



__all__ = ["str2datasets"]
