import json

json_config = {
    "part": int(input("the part is: ")),
    "user_equipment_num": int(input("the user equipment num is: ")),
    "base_station_computing_ability": float(input("the base station computing ability is: ")),
    "bandwidth": float(input("the band width is: "))
}

json.dump(json_config, open("prt" + "_" + str(int(json_config["part"])) + "_" +
                            "uen" + "_" + str(int(json_config["user_equipment_num"])) + "_" +
                            "bsa" + "_" + str(int(json_config["base_station_computing_ability"])) + "_" +
                            "bdw" + "_" + str(int(json_config["bandwidth"])) + ".json",
                            "w", encoding="utf-8"), indent=2)
