## icl.py
## Construct demos for in-context learning

from typing import Union, List
import numpy as np


class ICL_Demos(object):
    
    def __init__(self,
                 demos: List[dict] = [],
                 kshot: int = 1,
                 ):
        """
        Initializes an instance of the In-Context Learning.
        
        Parameters
        ----------
        demos : List[dict], optional
            List of demonstrations. 
        kshot : int, optional
            The number of examples for in-context learning.

        """
        self.demos = demos
        self.kshot = kshot
        self.N = len(demos)


    def generate(self,
                 method: str = "random",
                 ):
        """
        Generates a subset of demonstrations for in-context learning.
    
        Parameters
        ----------
        method : str, optional
            The method used for generating the subset.
    
        Raises
        ------
        ValueError
            If there are no demonstrations for in-context learning.
        NotImplementedError
            If the specified generation method is not implemented.
    
        Returns
        -------
        subset : TYPE
            The generated subset of demonstrations.
        """
        if self.N <= 0:
            raise ValueError("No demonstrations for ICL")
        # generation function
        subset = None
        if method == "random":
            subset = self._random_sample()
        else:
            raise NotImplementedError(f"Generation method for ICL {method} has not been implemented yet")
        return subset


    def _random_sample(self):
        """
        Samples a subset of demonstrations.

        Returns
        -------
        subset : List[dict]
            The randomly selected subset of demonstrations.
        """
        inds = np.random.choice(self.N, self.kshot, replace=False)
        subset = [self.demos[idx] for idx in list(inds)]
        return subset


if __name__ == '__main__':
    demos = [{'key':1}, {'del':2}, {'rel':3}]
    kshot = 1
    icl = ICL_Demos(demos, kshot)
    print(icl.generate())
