import pandas as pd
from os import path as osp

cutting_req_file = r'c:\tmp\work1\cutting_requirements.xlsx'
cutting_req_df = pd.read_excel(cutting_req_file)

df_targ = cutting_req_df[cutting_req_df['Type'] != 'File'].copy()

df_targ['Rotation'] =  -1
df_targ['Tilt'] = 0
df_targ['Mirror'] = 0
df_targ['IsSheet'] = 0

# Enforce data types
df_targ = df_targ.astype({'Quantity':int,
                          'Priority':int,
                          'Rotation':int,
                          'Tilt':int,
                          'Mirror':int,
                          'IsSheet':int})

# Append sheet (IsSheet = 1)
input_sheet_dxf = 'x'
# TODO:  Set lot and piece #, or change Name
new_key = len(df_targ)
df_targ.loc[new_key] = {'Type':'File',
                        'Name':'Lot NHxxx Piece x',
                        'Quantity':1,
                        'Priority':1,
                        'Rotation':-1,
                        'Tilt':0,
                        'Mirror':0,
                        'Value1':input_sheet_dxf,
                        'Value2':'',
                        'IsSheet':1}
input_csv = osp.join(r'c:\tmp','cutting_inputs_nestfab.csv')
df_targ.to_csv(input_csv, index=False)

