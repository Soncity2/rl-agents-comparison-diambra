# characters.py

GAME_CHARACTER_MAP = {
    "tektagt": {
        0: "Xiaoyu", 1: "Yoshimitsu", 2: "Nina", 3: "Law", 4: "Hwoarang",
        5: "Eddy", 6: "Paul", 7: "King", 8: "Lei", 9: "Jin",
        10: "Baek", 11: "Michelle", 12: "Armorking", 13: "Gunjack", 14: "Anna",
        15: "Brian", 16: "Heihachi", 17: "Ganryu", 18: "Julia", 19: "Jun",
        20: "Kunimitsu", 21: "Kazuya", 22: "Bruce", 23: "Kuma", 24: "Jack-Z",
        25: "Lee", 26: "Wang", 27: "P.Jack", 28: "Devil", 29: "True Ogre",
        30: "Ogre", 31: "Roger", 32: "Tetsujin", 33: "Panda", 34: "Tiger",
        35: "Angel", 36: "Alex", 37: "Mokujin", 38: "Unknown"
    }
    # You can add more games here.
}

# Inverse map: name → ID
GAME_CHARACTER_NAME_TO_ID = {
    game: {name: cid for cid, name in mapping.items()}
    for game, mapping in GAME_CHARACTER_MAP.items()
}

def id_to_character_name(game_id, character_id):
    return GAME_CHARACTER_MAP.get(game_id, {}).get(character_id, "Unknown")

def character_name_to_id(game_id, character_name):
    return GAME_CHARACTER_NAME_TO_ID.get(game_id, {}).get(character_name, -1)

def id_pair_to_names(game_id, character_id_1, character_id_2):
    return (
        id_to_character_name(game_id, character_id_1),
        id_to_character_name(game_id, character_id_2)
    )

def names_to_id_pair(game_id, name1, name2):
    return (
        character_name_to_id(game_id, name1),
        character_name_to_id(game_id, name2)
    )
