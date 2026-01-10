T_REPL = "to_replace"
REPL_W = "replace_with"

# Some general regext to apply to get first stage of clear text
GENERAL_CLEANERS_REGEX = {
    "repl_space_quot_space_to_comma": {
        T_REPL: r" \? ",
        REPL_W: ", ",
    },
    "repl_colon_comma_to_colon": {T_REPL: r":,", REPL_W: ":"},
    "repl_colon_quote_to_colon": {T_REPL: r": \?", REPL_W: ":"},
    "repl_quote_dot_to_dot": {T_REPL: r"\? \.", REPL_W: "."},
    "repl_quote_dot_to_dot2": {T_REPL: r"\?\.", REPL_W: "."},
    "repl_comma_dash_to_comma": {T_REPL: r", \-", REPL_W: ","},
    "repl_dd_comma_to_comma": {T_REPL: r", ,", REPL_W: ","},
    "repl_d_comma_to_comma": {T_REPL: r",,", REPL_W: ","},
}

# Abbreviations replacement
ABBREV_REPLACE_MAP = [
    ("np.", "np_"),
    ("ul.", "ul_"),
    ("tj.", "tj_"),
    ("inż.", "inż_"),
    ("S.A.", "S_A_"),
    ("s.a.", "s_a_"),
]

# www/url
REMOVE_URL = [
    (
        "((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]"
        "+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
        "WWW_URL",
    ),
]

# Add new line in some cases
ADD_NEW_LINE_IN_CASE = {
    "repl_beg_chat_num": {T_REPL: r"^\(\d+\)", REPL_W: ""},
    "repl_in_chat_num": {T_REPL: r"\(\d+\)", REPL_W: "\n"},
    "repl_dot_newline1": {
        T_REPL: r'(\s[A-zęąŃńćĆóÓźŹżŻśŚłŁ"()]+[0-9]{0,20}\.)',
        REPL_W: r"\1\n\n",
    },
    "repl_dot_newline2": {T_REPL: r"(\s[0-9]+\.)", REPL_W: r"\1\n\n"},
    "repl_dot_newline3": {T_REPL: r"(\s[A-Z]+\.)", REPL_W: r"\1\n\n"},
    "repl_char_eum_in": {T_REPL: r"\s[a-z]\)", REPL_W: "\n\n"},
}

# These characters may occur on the sentence end
SENTENCE_END_CHARS = "!@#$%^&*()_-+={[}];:'\"<,>./?"

# Enumerations
ENUMERATION_REGEX = {
    "enum_dot_standard": {T_REPL: r"(\s)[0-9]\.\s", REPL_W: r"\1"},
    "enum_round_brackets": {T_REPL: r"(\s)[0-9]\)\s", REPL_W: r"\1"},
}
