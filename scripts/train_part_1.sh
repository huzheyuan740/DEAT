echo "Train Part 1"
python ./main.py --json_name "prt_1_uen_10_bsa_10000000000_bdw_10000000" &
python ./main.py --json_name "prt_1_uen_20_bsa_20000000000_bdw_20000000" &
python ./main.py --json_name "prt_1_uen_30_bsa_30000000000_bdw_30000000" &
python ./main.py --json_name "prt_1_uen_40_bsa_40000000000_bdw_40000000" &
python ./main.py --json_name "prt_1_uen_50_bsa_50000000000_bdw_50000000" &
wait
