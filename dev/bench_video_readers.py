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
    # video_fpath = ub.grabdata('https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov')
    try:
        import vi3o
    except Exception:
        vi3o = None

    video_fpath = ub.grabdata('https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4')
    video_fpath = ub.grabdata('https://file-examples-com.github.io/uploads/2018/04/file_example_MOV_1280_1_4MB.mov')

    ti = timerit.Timerit(10, bestof=3, verbose=3, unit='ms')

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
    kwplot.autompl()

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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/ndsampler/dev/bench_video_readers.py
    """
    benchmark_video_readers()
