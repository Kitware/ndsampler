import timerit
import torch
import netharn as nh
import ubelt as ub
import networkx as nx
import torch.nn.functional as F


class CudaTimer(timerit.Timer):
    def toc(self):
        torch.cuda.synchronize()
        return super(CudaTimer, self).toc()


class CudaTimerit(timerit.Timerit):
    _default_timer_cls = CudaTimer


def _bench_index_select_put():
    B, C, H, W = (32, 100, 32, 32)
    class_energy = torch.rand(B, C, H, W)
    idxs = [2, 3, 5, 7, 11, 13, 17, 21]

    xpu = 1

    class_energy = class_energy.to(xpu)

    dim = 1

    ti = CudaTimerit(100, bestof=10, verbose=1)

    for timer in ti.reset('select multi-index (torch.index_select)'):
        with timer:
            index = torch.LongTensor(idxs).to(class_energy.device)
            select_result1 = torch.index_select(class_energy, dim=dim, index=index)

    for timer in ti.reset('select multi-index (fancy_index)'):
        with timer:
            fancy_prefix = [slice(None)] * dim
            fancy_index = tuple(fancy_prefix + [idxs])
            select_result2 = class_energy[fancy_index]

    for timer in ti.reset('select multi-index (select-stack)'):
        with timer:
            fancy_prefix = [slice(None)] * dim
            fancy_index = tuple(fancy_prefix + [idxs])
            select_result3 = [class_energy.select(dim, idx) for idx in idxs]
            select_result3 = torch.stack(select_result3, dim=dim)

    for timer in ti.reset('select multi-index (select-nostack)'):
        with timer:
            fancy_prefix = [slice(None)] * dim
            fancy_index = tuple(fancy_prefix + [idxs])
            select_result4_raw = [class_energy.select(dim, idx) for idx in idxs]
    select_result4 = torch.stack(select_result4_raw, dim=dim)

    assert torch.all(select_result2 == select_result1).cpu().numpy()
    assert torch.all(select_result3 == select_result1).cpu().numpy()
    assert torch.all(select_result4 == select_result1).cpu().numpy()


def _bench_put():
    B, C, H, W = (32, 100, 32, 32)
    idxs = [2, 3, 5, 7, 11, 13, 17, 21]

    B, C, H, W = (32, 100, 64, 64)
    idxs = [2, 3, 5, 7, 11, 13, 17, 21]

    # B, C, H, W = (1, 7, 3, 3)
    # idxs = [2, 3, 5]

    dim = 1
    class_energy = torch.rand(B, C, H, W)
    class_energy = class_energy.to(nh.XPU.cast('auto').main_device)

    ti = CudaTimerit(1000, bestof=100, verbose=1)
    outputs = ub.odict()

    for timer in ti.reset('put multi-index (fancy_index)'):
        class_logprobs = torch.zeros_like(class_energy)
        with timer:
            fancy_prefix = [slice(None)] * dim
            fancy_index = tuple(fancy_prefix + [idxs])
            index = torch.LongTensor(idxs).to(class_energy.device)
            selected = torch.index_select(class_energy, dim=dim, index=index)
            class_logprobs[fancy_index] = selected
    outputs[ti.label] = class_logprobs.clone()
    assert not torch.all(class_logprobs[fancy_index] == 0)

    for timer in ti.reset('put multi-index (loop, select)'):
        class_logprobs = torch.zeros_like(class_energy)
        with timer:
            for idx in idxs:
                class_logprobs.select(dim, idx)[:] = class_energy.select(dim, idx)
    outputs[ti.label] = class_logprobs.clone()
    assert torch.all(class_logprobs.select(dim, idx) == class_energy.select(dim, idx))

    for timer in ti.reset('index-copy multi-index'):
        class_logprobs = torch.zeros_like(class_energy)
        with timer:
            index = torch.LongTensor(idxs).to(class_energy.device)
            selected = torch.index_select(class_energy, dim=dim, index=index)
            class_logprobs.index_copy_(dim, index, selected)
    outputs[ti.label] = class_logprobs.clone()

    for k1, k2 in ub.iter_window(outputs, 2):
        if torch.all(outputs[k1] == outputs[k2]):
            print('MATCH: k1={} matches k2={}'.format(k1, k2))
        else:
            print('DIFF: k1={} DIFFERS k2={}'.format(k1, k2))
            print((torch.abs(outputs[k1] - outputs[k2])).sum())

    # for timer in ti.reset('put multi-index 1 (fancy_index)'):
    #     class_logprobs = torch.zeros_like(class_energy)
    #     with timer:
    #         fancy_prefix = [slice(None)] * dim
    #         fancy_index = tuple(fancy_prefix + [idxs])
    #         class_logprobs[fancy_index] = class_logprobs[fancy_index] + 1
    # outputs[ti.label] = class_logprobs.clone()

    # assert not torch.all(class_logprobs[fancy_index] == 0)

    # for timer in ti.reset('put multi-index 1 (loop, select)'):
    #     class_logprobs = torch.zeros_like(class_energy)
    #     with timer:
    #         for idx in idxs:
    #             class_logprobs.select(dim, idx)[:] = class_logprobs.select(dim, idx) + 1
    # assert torch.all(class_logprobs.select(dim, idx) == 1)
    # outputs[ti.label] = class_logprobs.clone()

    for timer in ti.reset('index-copy multi-index (just-copy)'):
        class_logprobs = torch.zeros_like(class_energy)
        index = torch.LongTensor(idxs).to(class_energy.device)
        selected = torch.index_select(class_energy, dim=dim, index=index)
        with timer:
            class_logprobs.index_copy_(dim, index, selected)
    outputs[ti.label] = class_logprobs.clone()

    for timer in ti.reset('put multi-index (fancy_index) (just-copy)'):
        class_logprobs = torch.zeros_like(class_energy)
        fancy_prefix = [slice(None)] * dim
        fancy_index = tuple(fancy_prefix + [idxs])
        index = torch.LongTensor(idxs).to(class_energy.device)
        selected = torch.index_select(class_energy, dim=dim, index=index)
        with timer:
            class_logprobs[fancy_index] = selected
    outputs[ti.label] = class_logprobs.clone()
    assert not torch.all(class_logprobs[fancy_index] == 0)


def _bench_catgraph_conditional_log_softmax_solution():
    from ovharn import category_tree

    graph = nx.generators.gnr_graph(30, 0.3, seed=321).reverse()
    self = category_tree.CategoryTree(graph)
    class_energy = torch.randn(16, len(self.idx_to_node), 15, 15)
    class_energy = class_energy.to(nh.XPU.cast('auto').main_device)
    dim = 1

    def method1(self, class_energy, dim):
        cond_logprobs = torch.empty_like(class_energy)
        # Move indexes onto the class_energy device (perhaps precache this)
        index_groups = [torch.LongTensor(idxs).to(class_energy.device)
                        for idxs in self.idx_groups]
        for index in index_groups:
            # Take each subset of classes that are mutually exclusive
            energy_group = torch.index_select(class_energy, dim=dim, index=index)
            # Then apply the log_softmax to those sets
            logprob_group = F.log_softmax(energy_group, dim=dim)
            cond_logprobs.index_copy_(dim, index, logprob_group)
        return cond_logprobs

    def method2(self, class_energy, dim):
        cond_logprobs = torch.empty_like(class_energy)
        fancy_prefix = [slice(None)] * dim
        for idxs in self.idx_groups:
            fancy_index = tuple(fancy_prefix + [idxs])
            cond_logprobs[fancy_index] = F.log_softmax(class_energy[fancy_index], dim=dim)
        return cond_logprobs

    ti = CudaTimerit(500, bestof=50, verbose=1, unit='us')
    outputs = ub.odict()
    for timer in ti.reset('method1'):
        with timer:
            cond_logprobs = method1(self, class_energy, dim)
    outputs[ti.label] = cond_logprobs.clone()

    for timer in ti.reset('method2'):
        with timer:
            cond_logprobs = method2(self, class_energy, dim)
    outputs[ti.label] = cond_logprobs.clone()

    # ------

    for k1, k2 in ub.iter_window(outputs, 2):
        if torch.all(outputs[k1] == outputs[k2]):
            print('MATCH: k1={} matches k2={}'.format(k1, k2))
        else:
            print('DIFF: k1={} DIFFERS k2={}'.format(k1, k2))
            print((torch.abs(outputs[k1] - outputs[k2])).sum())

    # Meausre how much overhead creating the LongTensor takes.
    # Is it worth precaching? Not really, its like 1% of the time.
    def method1_overhead(self, class_energy, dim):
        # Move indexes onto the class_energy device (perhaps precache this)
        index_groups = [torch.LongTensor(idxs).to(class_energy.device)
                        for idxs in self.idx_groups]
        for index in index_groups:
            pass
    for timer in ti.reset('method1-overhead'):
        with timer:
            method1_overhead(self, class_energy, dim)


def _bench_catgraph_sink_log_softmax():
    from ovharn import category_tree
    sink_nodes = category_tree.sink_nodes

    def sink_log_softmax_method1(self, class_energy, dim):
        leaf_idxs = sorted(self.node_to_idx[node]
                           for node in sink_nodes(self.graph))
        class_logprobs = torch.empty_like(class_energy)

        fancy_prefix = [slice(None)] * dim
        fancy_index = tuple(fancy_prefix + [leaf_idxs])
        class_logprobs[fancy_index] = F.log_softmax(class_energy[fancy_index], dim=dim)

        @ub.memoize
        def populate1(node):
            """ dynamic program to compute absolute class log probability """
            children = list(self.graph.successors(node))
            child_idxs = sorted(self.node_to_idx[node] for node in children)
            if len(children) > 0:
                # Ensure that all children are populated before the parents
                for child in children:
                    populate1(child)
                node_idx = self.node_to_idx[node]
                fancy_node_index = tuple(fancy_prefix + [node_idx])
                fancy_children_index = tuple(fancy_prefix + [child_idxs])
                class_logprobs[fancy_node_index] = torch.logsumexp(class_logprobs[fancy_children_index], dim=dim)
        for node in self.graph.nodes():
            populate1(node)
        return class_logprobs

    def sink_log_softmax_method2(self, class_energy, dim):
        class_logprobs = torch.empty_like(class_energy)
        leaf_idxs = sorted(self.node_to_idx[node]
                           for node in sink_nodes(self.graph))
        leaf_idxs = torch.LongTensor(leaf_idxs).to(class_energy.device)

        leaf_energy = torch.index_select(class_energy, dim=dim,
                                         index=leaf_idxs)
        leaf_logprobs = F.log_softmax(leaf_energy, dim=dim)
        class_logprobs.index_copy_(dim, leaf_idxs, leaf_logprobs)

        @ub.memoize
        def populate2(node):
            """ dynamic program to compute absolute class log probability """
            children = list(self.graph.successors(node))
            if len(children) > 0:
                # Ensure that all children are populated before the parents
                for child in children:
                    populate2(child)
                child_idxs = sorted(self.node_to_idx[node] for node in children)
                child_idxs = torch.LongTensor(child_idxs).to(class_energy.device)
                node_idx = self.node_to_idx[node]
                selected = torch.index_select(class_logprobs, dim=dim,
                                              index=child_idxs)
                total = torch.logsumexp(selected, dim=dim)
                class_logprobs.select(dim, node_idx)[:] = total

        for node in self.graph.nodes():
            populate2(node)
        return class_logprobs

    from ovharn import category_tree
    graph = nx.generators.gnr_graph(30, 0.3, seed=321).reverse()
    self = category_tree.CategoryTree(graph)
    class_energy = torch.randn(16, len(self.idx_to_node), 15, 15)
    class_energy = class_energy.to(nh.XPU.cast('auto').main_device)
    dim = 1

    ti = CudaTimerit(500, bestof=50, verbose=1, unit='us')
    outputs = ub.odict()
    for timer in ti.reset('method1'):
        with timer:
            cond_logprobs = sink_log_softmax_method1(self, class_energy, dim)
    outputs[ti.label] = cond_logprobs.clone()

    for timer in ti.reset('method2'):
        with timer:
            cond_logprobs = sink_log_softmax_method2(self, class_energy, dim)
    outputs[ti.label] = cond_logprobs.clone()

    for k1, k2 in ub.iter_window(outputs, 2):
        if torch.all(outputs[k1] == outputs[k2]):
            print('MATCH: k1={} matches k2={}'.format(k1, k2))
        else:
            print('DIFF: k1={} DIFFERS k2={}'.format(k1, k2))
            print((torch.abs(outputs[k1] - outputs[k2])).sum())
