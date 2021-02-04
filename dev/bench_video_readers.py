import pandas as pd
import ubelt as ub
import cv2
import kwarray
import timerit


class CV2VideoReader(ub.NiceRepr):
    def __init__(self, fpath):
        self.fpath = fpath
        self._cap = cv2.VideoCapture(fpath)
        self._len = None

    def tell(self):
        index = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        return index

    def seek(self, index):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    def __del__(self):
        self._cap.release()

    def meta(self):
        keys = [n for n in dir(cv2) if n.startswith('CAP_PROP_')]
        meta = {k: self._cap.get(getattr(cv2, k)) for k in keys}
        return meta

    def __len__(self):
        if self._len is None:
            self._len = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._len

    def __iter__(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def __getitem__(self, index):
        self.seek(index)
        ret, frame = self._cap.read()
        if not ret:
            raise IndexError(index)
        return frame


def benchmark_video_readers():
    """
    "On My Machine" I get:

        ti.measures = {
            'mean'    : {
                'cv2 sequential access'          : 0.0137,
                'decord sequential access'       : 0.0175,
                'cv2 open + first access'        : 0.0222,
                'decord open + first access'     : 0.0565,
                'vi3o sequential access'         : 0.0642,
                'cv2 open + one random access'   : 0.0723,
                'decord open + one random access': 0.0946,
                'vi3o open + first access'       : 0.1045,
                'cv2 random access'              : 0.3316,
                'decord random access'           : 0.3472,
                'decord random batch access'     : 0.3482,
                'vi3o open + one random access'  : 0.3590,
                'vi3o random access'             : 1.6660,
            },
            'mean+std': {
                'cv2 sequential access'          : 0.0145,
                'decord sequential access'       : 0.0182,
                'cv2 open + first access'        : 0.0230,
                'vi3o sequential access'         : 0.0881,
                'decord open + first access'     : 0.1038,
                'vi3o open + first access'       : 0.1059,
                'cv2 open + one random access'   : 0.1151,
                'decord open + one random access': 0.1329,
                'cv2 random access'              : 0.3334,
                'decord random access'           : 0.3496,
                'decord random batch access'     : 0.3511,
                'vi3o open + one random access'  : 0.5215,
                'vi3o random access'             : 1.6890,
            },
            'mean-std': {
                'decord open + first access'     : 0.0091,
                'cv2 sequential access'          : 0.0130,
                'decord sequential access'       : 0.0168,
                'cv2 open + first access'        : 0.0214,
                'cv2 open + one random access'   : 0.0295,
                'vi3o sequential access'         : 0.0403,
                'decord open + one random access': 0.0563,
                'vi3o open + first access'       : 0.1032,
                'vi3o open + one random access'  : 0.1965,
                'cv2 random access'              : 0.3299,
                'decord random access'           : 0.3448,
                'decord random batch access'     : 0.3452,
                'vi3o random access'             : 1.6429,
            },
            'min'     : {
                'cv2 sequential access'          : 0.0128,
                'decord sequential access'       : 0.0166,
                'cv2 open + first access'        : 0.0210,
                'vi3o sequential access'         : 0.0233,
                'decord open + first access'     : 0.0251,
                'cv2 open + one random access'   : 0.0282,
                'decord open + one random access': 0.0527,
                'vi3o open + one random access'  : 0.1013,
                'vi3o open + first access'       : 0.1026,
                'cv2 random access'              : 0.3299,
                'decord random access'           : 0.3433,
                'decord random batch access'     : 0.3452,
                'vi3o random access'             : 1.6423,
            },
        }

    """
    # video_fpath = ub.grabdata('https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov')
    try:
        import vi3o
    except Exception:
        vi3o = None

    video_fpath = ub.grabdata('https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4')
    video_fpath = ub.grabdata('https://file-examples-com.github.io/uploads/2018/04/file_example_MOV_1280_1_4MB.mov')

    ti = timerit.Timerit(2, bestof=2, verbose=3, unit='ms')

    video_length = len(CV2VideoReader(video_fpath))
    num_frames = min(5, video_length)
    rng = kwarray.ensure_rng(0)
    random_indices = rng.randint(0, video_length, size=num_frames).tolist()

    if True:
        with timerit.Timer(label='open cv2') as cv2_open_timer:
            cv2_video = CV2VideoReader(video_fpath)

        for timer in ti.reset('cv2 sequential access'):
            cv2_video.seek(0)
            with timer:
                for frame, _ in zip(cv2_video, range(num_frames)):
                    pass

        for timer in ti.reset('cv2 random access'):
            with timer:
                for index in random_indices:
                    cv2_video[index]

    if vi3o is not None:
        with timerit.Timer(label='open vi3o') as vi3o_open_timer:
            vi3o_video = vi3o.Video(video_fpath)

        for timer in ti.reset('vi3o sequential access'):
            with timer:
                for frame, _ in zip(vi3o_video, range(num_frames)):
                    pass

        for timer in ti.reset('vi3o random access'):
            with timer:
                for index in random_indices:
                    vi3o_video[index]

    if True:
        import decord
        with timerit.Timer(label='open decord') as decord_open_timer:
            decord_video = decord.VideoReader(video_fpath)

        for timer in ti.reset('decord sequential access'):
            with timer:
                for frame, _ in zip(decord_video, range(num_frames)):
                    pass

        for timer in ti.reset('decord random access'):
            with timer:
                for index in random_indices:
                    decord_video[index]

        for timer in ti.reset('decord random batch access'):
            with timer:
                decord_video.get_batch(random_indices)

    if True:
        # One Random Access Case

        def _work_to_clear_io_caches():
            import kwimage
            # Let some caches be cleared
            for i in range(10):
                for key in kwimage.grab_test_image.keys():
                    kwimage.grab_test_image(key)

        rng = kwarray.ensure_rng(0)
        for timer in ti.reset('cv2 open + one random access'):
            _work_to_clear_io_caches()
            with timer:
                _cv2_video = CV2VideoReader(video_fpath)
                index = rng.randint(0, video_length, size=1)[0]
                _cv2_video[index]

        if vi3o is not None:
            rng = kwarray.ensure_rng(0)
            for timer in ti.reset('vi3o open + one random access'):
                _work_to_clear_io_caches()
                with timer:
                    _vi3o_video = vi3o.Video(video_fpath)
                    index = rng.randint(0, video_length, size=1)[0]
                    _vi3o_video[index]

        rng = kwarray.ensure_rng(0)
        for timer in ti.reset('decord open + one random access'):
            _work_to_clear_io_caches()
            with timer:
                _decord_video = decord.VideoReader(video_fpath)
                index = rng.randint(0, video_length, size=1)[0]
                _decord_video[index]

        for timer in ti.reset('cv2 open + first access'):
            _work_to_clear_io_caches()
            with timer:
                _cv2_video = CV2VideoReader(video_fpath)
                _cv2_video[0]

        if vi3o is not None:
            for timer in ti.reset('vi3o open + first access'):
                _work_to_clear_io_caches()
                with timer:
                    _vi3o_video = vi3o.Video(video_fpath)
                    _vi3o_video[0]

        for timer in ti.reset('decord open + first access'):
            _work_to_clear_io_caches()
            with timer:
                _decord_video = decord.VideoReader(video_fpath)
                _decord_video[0]

    measures = ub.map_vals(ub.sorted_vals, ti.measures)
    print('ti.measures = {}'.format(ub.repr2(measures, nl=2, align=':', precision=4)))
    print('cv2_open_timer.elapsed    = {!r}'.format(cv2_open_timer.elapsed))
    print('decord_open_timer.elapsed = {!r}'.format(decord_open_timer.elapsed))
    if vi3o:
        print('vi3o_open_timer.elapsed   = {!r}'.format(vi3o_open_timer.elapsed))

    import kwplot
    import seaborn as sns
    sns.set()
    plt = kwplot.autoplt()

    df = pd.DataFrame(ti.measures)
    df['key'] = df.index
    df['expt'] = df['key'].apply(lambda k: ' '.join(k.split(' ')[1:]))
    df['module'] = df['key'].apply(lambda k: k.split(' ')[0])

    # relmod = 'decord'
    relmod = 'cv2'
    for k, group in df.groupby('expt'):
        measure = 'mean'
        relval = group[group['module'] == relmod][measure].values.ravel()
        if len(relval) > 0:
            assert len(relval) == 1
            df.loc[group.index, measure + '_rel'] = group[measure] / relval
            df.loc[group.index, measure + '_slower_than_' + relmod] = group[measure] / relval
            df.loc[group.index, measure + '_faster_than_' + relmod] = relval / group[measure]

    fig = kwplot.figure(fnum=1, doclf=True)
    ax = fig.gca()
    y_key = "mean_faster_than_" + relmod

    sub_df = df.loc[~df[y_key].isnull()]
    sns.barplot(
        x="expt", y=y_key, data=sub_df, hue='module', ax=ax)
    ax.set_title('cpu video reading benchmarks')

    plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/dev/bench_video_readers.py
    """
    benchmark_video_readers()
