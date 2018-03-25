from util import connect, static_data_dir
import os


if __name__ == "__main__":
    db = connect()
    out_dir = os.path.join(static_data_dir, 'static_feature')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    

    download_

