from packages.sr_downloader.downloader import Downloader

if __name__ == "__main__":
    d = Downloader()
    lis = d.get_all_osu_list("D:/Software/osu/Songs/1817483 YUI - again (TV Size)")
    print(lis)