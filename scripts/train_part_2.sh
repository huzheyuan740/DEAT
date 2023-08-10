echo "Train Part 2"
python ./main.py --json_name "prt_2_uen_10_bsa_25000000000_bdw_25000000" &
python ./main.py --json_name "prt_2_uen_20_bsa_25000000000_bdw_25000000" &
python ./main.py --json_name "prt_2_uen_30_bsa_25000000000_bdw_25000000" &
python ./main.py --json_name "prt_2_uen_40_bsa_25000000000_bdw_25000000" &
python ./main.py --json_name "prt_2_uen_50_bsa_25000000000_bdw_25000000" &
wait
