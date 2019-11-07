
def qtree_mwe():
    import numpy as np
    import pyqtree

    # Populate a qtree with a set of random boxes
    aid_to_tlbr = {779: np.array([412, 404, 488, 455]),
                   781: np.array([127, 429, 194, 517]),
                   782: np.array([459, 282, 517, 364]),
                   784: np.array([404, 160, 496, 219]),
                   785: np.array([336, 178, 367, 209]),
                   786: np.array([366, 459, 451, 527]),
                   788: np.array([491, 434, 532, 504]),
                   789: np.array([251, 185, 322, 248]),
                   790: np.array([266, 104, 387, 162]),
                   791: np.array([ 65, 296, 138, 330]),
                   792: np.array([331, 241, 368, 347])}
    orig_qtree = pyqtree.Index((0, 0, 600, 600))
    for aid, tlbr in aid_to_tlbr.items():
        orig_qtree.insert(aid, tlbr)

    # Issue a query and inspect results
    query = np.array([0, 0, 300, 300])
    original_result = orig_qtree.intersect(query)

    # We see that everything looks fine
    print('original_result = {!r}'.format(sorted(original_result)))

    # Serialize and unserialize the Index, and inspect results
    import pickle
    serial = pickle.dumps(orig_qtree)
    new_qtree = pickle.loads(serial)

    # Issue the same query on the reloaded Index, the result now
    # contains duplicate items!!
    new_result = new_qtree.intersect(query)
    print('new_result = {!r}'.format(sorted(new_result)))

    ####
    # Experiments
    ####

    # Question: Does serializing a second time have any effect?
    # Ans: No
    if True:
        third_qtree = pickle.loads(pickle.dumps(new_qtree))
        third_result = third_qtree.intersect(query)
        print('third_result = {!r}'.format(sorted(third_result)))

    # Question: What if we use smaller node ids?
    # Ans: No
    if True:
        aid_to_tlbr = {0: np.array([412, 404, 488, 455]),
                       1: np.array([127, 429, 194, 517]),
                       2: np.array([459, 282, 517, 364]),
                       3: np.array([404, 160, 496, 219]),
                       4: np.array([336, 178, 367, 209]),
                       5: np.array([366, 459, 451, 527]),
                       6: np.array([491, 434, 532, 504]),
                       7: np.array([251, 185, 322, 248]),
                       8: np.array([266, 104, 387, 162]),
                       9: np.array([ 65, 296, 138, 330]),
                       10: np.array([331, 241, 368, 347])}
        qtree3 = pyqtree.Index((0, 0, 600, 600))
        for aid, tlbr in aid_to_tlbr.items():
            qtree3.insert(aid, tlbr)
        query = np.array([0, 0, 300, 300])
        result3 = qtree3.intersect(query)
        print('result3 = {!r}'.format(sorted(result3)))
        qtree4 = pickle.loads(pickle.dumps(qtree3))
        result4 = qtree4.intersect(query)
        print('result4 = {!r}'.format(sorted(result4)))


def iter_qtree(self):
    """
    ub.find_duplicates(list(iter_qtree(qtree)))
    """
    for node in self.nodes:
        yield node.item
    for quad in self:
        for node in quad.nodes:
            yield node.item
