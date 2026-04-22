from icrawler.builtin import BingImageCrawler
import os

os.chdir(f"{os.path.expanduser('~')}/AI-and-Deep-Learning-Group-8--6165/")

os.makedirs('./dataset/backgrounds/grass', exist_ok = True)
os.makedirs('./dataset/backgrounds/soil', exist_ok = True)
os.makedirs('./dataset/backgrounds/garden', exist_ok = True)

crawler = BingImageCrawler(storage={'root_dir': './dataset/backgrounds/grass'})
crawler.crawl(keyword='grass ground texture', max_num=100)

crawler = BingImageCrawler(storage={'root_dir': './dataset/backgrounds/soil'})
crawler.crawl(keyword='soil ground texture', max_num=100)

crawler = BingImageCrawler(storage={'root_dir': './dataset/backgrounds/garden'})
crawler.crawl(keyword='garden texture', max_num=100)
