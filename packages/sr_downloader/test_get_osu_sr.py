from packages.sr_downloader.downloader import Downloader

if __name__ == "__main__":
    d = Downloader()
    lis = d.get_osu_sr()
    # print(lis)