      
# activate Environment
.\pan25-detox-env\Scripts\activate  

# give comand
(pan25-detox-env) PS C:\Users\jonaz\git\Maria_stuff\Maria_code\baseline_backtranslation> python .\main-clean.py `     
>>   --input_path .\paradetox_en.tsv ` 
>>   --output_path .\backtranslation_output_en.tsv `     
>>   --batch_size 4 `
>>   --device cuda `
>>   --verbose True

python .\main-refined.py `     
--input_path .\paradetox_en.tsv ` 
--output_path .\refined_output_en.tsv `     
--batch_size 4 `
--device cuda `
--verbose True

python .\main-refined.py `
  --input_path .\paradetox_en.tsv `
  --output_path .\refined_output_en.tsv `
  --batch_size 4 `
  --device cuda `
  --verbose True