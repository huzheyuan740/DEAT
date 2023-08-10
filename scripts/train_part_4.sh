echo "Train Part 4"
python ./main.py --json_name "prt_4_uen_25_bsa_25000000000_bdw_10000000" &
python ./main.py --json_name "prt_4_uen_25_bsa_25000000000_bdw_20000000" &
python ./main.py --json_name "prt_4_uen_25_bsa_25000000000_bdw_30000000" &
python ./main.py --json_name "prt_4_uen_25_bsa_25000000000_bdw_40000000" &
python ./main.py --json_name "prt_4_uen_25_bsa_25000000000_bdw_50000000" &
wait
