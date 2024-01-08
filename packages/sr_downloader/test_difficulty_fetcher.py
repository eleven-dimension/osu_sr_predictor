from packages.sr_downloader.downloader import Downloader, DifficultyAPIFetcher

if __name__ == "__main__":
    d = DifficultyAPIFetcher()
    sr = d.fetch_difficulty(1087294)
    print(sr)