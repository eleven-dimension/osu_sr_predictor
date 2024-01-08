from packages.osu_file_analyzer.analyzer import OsuFileAnalyzer

if __name__ == "__main__":
    a = OsuFileAnalyzer()
    # a.analyze_map("./data/milk.osu")
    # print(a.is_mania_4k("./data/milk.osu"))
    # print(a.is_mania_4k("./data/7k.osu"))
    print(a.get_beatmap_id("./data/milk.osu"))