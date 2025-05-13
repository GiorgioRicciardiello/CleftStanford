"""
The raw files are in .ASC structure and uses the .DO file for indicating which columns and how many characters we must
select
"""

import re
import pandas as pd
import pathlib

class ASCParser:
    def __init__(self,
                 do_file_path:pathlib.Path,
                 do_file_output_path:pathlib.Path,
                 asc_file_path:pathlib.Path,
                 asc_file_output_path:pathlib.Path,
                 ):
        self.do_file_path = do_file_path
        self.do_file_output_path = do_file_output_path
        self.asc_file_path = asc_file_path
        self.asc_file_output_path = asc_file_output_path

        self.colspecs = []
        self.colnames = []
        self.labels = {}

    def parse_do_file(self):
        with open(self.do_file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            infix_match = re.match(r'.*\s+(\w+)\s+(\d+)-\s*(\d+)', line)
            label_match = re.match(r'label var (\w+)\s+"(.+?)"', line)

            if infix_match:
                varname = infix_match.group(1)
                start = int(infix_match.group(2)) - 1  # Convert to 0-index
                end = int(infix_match.group(3))
                self.colnames.append(varname)
                self.colspecs.append((start, end))

            if label_match:
                varname = label_match.group(1)
                description = label_match.group(2)
                self.labels[varname] = description

    def _to_dataframe(self, asc_file_path):
        df = pd.read_fwf(asc_file_path, colspecs=self.colspecs, names=self.colnames)
        return df

    def save_asc_to_csv(self) -> pd.DataFrame:
        df = self._to_dataframe(self.asc_file_path)
        df.to_csv(self.asc_file_output_path, index=False)
        return df

    def save_do_as_csv(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.labels.items()), columns=['code', 'label'])
        df.to_csv(self.do_file_output_path, index=False)
        return df


# class ASCParser:
#     def __init__(self,
#                  do_file_path: Union[str, pathlib.Path],
#                  asc_file_path: Union[str, pathlib.Path]):
#         self.do_file_path = Path(do_file_path) if isinstance(do_file_path, str) else do_file_path
#         self.asc_file_path = Path(asc_file_path) if isinstance(asc_file_path, str) else asc_file_path
#         self.colspecs = []
#         self.colnames = []
#         self.labels = {}
#
#     def _parse_do_file(self):
#         with open(self.do_file_path, 'r') as file:
#             lines = file.readlines()
#
#         # Parse variable names and colspecs
#         var_pattern = re.compile(r'\s*(long|int|byte|double|str)\s+(\w+)\s+(\d+)-\s*(\d+)')
#         for line in lines:
#             match = var_pattern.search(line)
#             if match:
#                 _, var_name, start, end = match.groups()
#                 self.colnames.append(var_name)
#                 self.colspecs.append((int(start) - 1, int(end)))
#
#         # Parse labels
#         label_pattern = re.compile(r'label var (\w+)\s+"(.+)"')
#         for line in lines:
#             match = label_pattern.search(line)
#             if match:
#                 var, label = match.groups()
#                 self.labels[var] = label
#
#     def read_asc_file(self):
#         self._parse_do_file()
#         df = pd.read_fwf(self.asc_file_path, colspecs=self.colspecs, names=self.colnames)
#         return df
#
#     def export_to_csv(self, output_path: str):
#         df = self.read_asc_file()
#         df.to_csv(output_path, index=False)
#         return output_path