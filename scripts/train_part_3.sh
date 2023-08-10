echo "Train Part 3"
python ./main.py --json_name "prt_3_uen_25_bsa_10000000000_bdw_25000000" &
python ./main.py --json_name "prt_3_uen_25_bsa_20000000000_bdw_25000000" &
python ./main.py --json_name "prt_3_uen_25_bsa_30000000000_bdw_25000000" &
python ./main.py --json_name "prt_3_uen_25_bsa_40000000000_bdw_25000000" &
python ./main.py --json_name "prt_3_uen_25_bsa_50000000000_bdw_25000000" &
wait
