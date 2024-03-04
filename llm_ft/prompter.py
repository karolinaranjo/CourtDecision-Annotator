"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union, List


class Prompter(object):

    def __init__(self,
                 kshot: int = 0,
                 verbose: bool = False,
                 cutoff_length: int = 128,
                 ):
        """
       Initializes the prompter object.

       Parameters
       ----------
       kshot : int, optional
           Number of shots for few-shot learning.
       verbose : bool, optional
           Whether to print verbose output.
       cutoff_length : int, optional
           Maximum length of generated prompts.
           
       Raises
       ------
       ValueError
           If the specified template file cannot be read.
        """
        self.verbose = verbose
        self.kshot = kshot
        self.cutoff_length = cutoff_length
        # Prompt template
        file_name = None
        if kshot > 0:
            file_name = osp.join("templates", "kshot.json")
        else:
            file_name = osp.join("templates", "base.json")
            
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")

        with open(file_name) as fp:
            self.template = json.load(fp)
        
        if self.verbose:
            print(
                f"Using prompt template {file_name}: {self.template['description']}"
            )


    
    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            output: Union[None, str] = None,
            demos: Union[None, List[dict]] = None,
            cutoff_length: int = 128,
    ) -> str:
        """
        Generates a prompt based on the provided components.
    
        Parameters
        ----------
        instruction : str
              Instruction for the task.
        input : Union[None, str], optional
             Input for the task
          output : Union[None, str], optional
              Expected output for the task.
          demos : Union[None, List[dict]], optional
              Demonstrations for in-context learning.
          cutoff_length : int, optional
              Maximum length of the generated prompt. The default is 128.
        
          Raises
          ------
          KeyError
              If no input value is provided for the task.
          ValueError
              If there are no demonstrations for in-context learning or 
              if the number of examples for in-context learning is negative.
        
          Returns
          -------
          str
              Generated prompt.

        """
        if input is None:
            raise KeyError(f"No input value for the task")
        
        # truncate it
        items = input.split()
        if len(items) > cutoff_length:
            input = " ".join(items[:cutoff_length])
        
        if self.kshot == 0:
            res = self.template["prompt_input"].format(
                instruction=instruction,
                input=input,
            )
        elif self.kshot > 0:
            # In-context learning
            if demos is None:
                raise ValueError("No demostrations for ICL")
            
            # Instruction part
            res = self.template["instruction"].format(
                instruction = instruction,
            )
            
            # Demonstration part
            for item in demos:
                item_input = item["demo_part"].split('\n\n### Response:\n')
                tokens = item_input[0][len('\n\n### Input:\n'):].split()
                output = item_input[1]
                if len(tokens) > cutoff_length:
                    item_input = " ".join(tokens[:cutoff_length])
                demo_text = self.template["demo_part"].format(
                    input = item_input,
                    output = output,
                )
                res = f"{res}{demo_text}"
            # Query (input) part
            query_text = self.template["query_part"].format(
                input=input,
            )
            res = f"{res}{query_text}"
        else:
            raise ValueError(f"The number of examples for ICL cannot be negative: {self.kshot}")

        # print(f"Constructed result: {res}")
        if output:
            res = f"{res}{output}" # Pair the input with output
        if self.verbose:
            print(res)
        return res
    

    def generate_seq2seq_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            output: Union[None, str] = None,
            demos: Union[None, List[dict]] = None,
            cutoff_length: int = 128,
    ) -> str:
        """
            
        Generates a prompt for sequence-to-sequence tasks.
    
        Parameters
        ----------
        instruction : str
            Instruction for the task.
        input : Union[None, str], optional
            Input for the task. 
        output : Union[None, str], optional
            Expected output for the task.
        demos : Union[None, List[dict]], optional
            Demonstrations for in-context learning.
        cutoff_length : int, optional
            Maximum length of the generated prompt. The default is 128.
    
        Raises
        ------
        KeyError
            If no input value is provided for the task.
        ValueError
            If there are no demonstrations for in-context learning or
            if the number of examples for in-context learning is negative.
    
        Returns
        -------
        Dict[str, str]
            Dictionary containing the inputs and labels of the generated prompt.

        """
        if input is None:
            raise KeyError(f"No input value for the task")
        
        # truncate it
        items = input.split()
        if len(items) > cutoff_length:
            input = " ".join(items[:cutoff_length])
        
        if self.kshot == 0:
            res = self.template["prompt_input"].format(
                instruction=instruction,
                input=input,
            )
        elif self.kshot > 0:
            # In-context learning
            if demos is None:
                raise ValueError("No demostrations for ICL")
            
            # Instruction part
            res = self.template["instruction"].format(
                instruction = instruction,
            )
            
            # Demonstration part
            for item in demos:
                item_input = item["demo_part"].split('\n\n### Response:\n')
                tokens = item_input[0][len('\n\n### Input:\n'):].split()
                output = item_input[1]
                if len(tokens) > cutoff_length:
                    item_input = " ".join(tokens[:cutoff_length])
                demo_text = self.template["demo_part"].format(
                    input = item_input,
                    output = output,
                )
                res = f"{res}{demo_text}"
            # Query (input) part
            query_text = self.template["query_part"].format(
                input=input,
            )
            res = f"{res}{query_text}"
        else:
            raise ValueError(f"The number of examples for ICL cannot be negative: {self.kshot}")

        # print(f"Constructed result: {res}")
        # if output:
        #     res = f"{res}{output}" # Pair the input with output
        if self.verbose:
            print(res)
        return {"inputs": res, "labels": output}
    

    def get_response(self, output: str) -> str:
        """
        Extracts the response from the model output.
    
        Parameters
        ----------
        output : str
            Output generated by the model.
    
        Raises
        ------
        ValueError
            If the output format is unrecognized.
    
        Returns
        -------
        str
            Extracted response from the output.

        """
        if self.template["response_split"] in output:
            # Get the last chuck, prepare for in-context learning
            return output.split(self.template["response_split"])[-1].strip()
        else:
            # raise ValueError(f"Unrecognized string: {output}")
            return output
