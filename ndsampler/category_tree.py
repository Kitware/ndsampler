# -*- coding: utf-8 -*-
"""
Notes from YOLO-9000:
    * perform multiple softmax operations over co-hyponyms
    * we compute the softmax over all sysnsets that are hyponyms of the same concept

    synsets - sets of synonyms (word or phrase that means exactly or nearly the same as another)

    hyponymn - a word of more specific meaning than a general or
        superordinate term applicable to it. For example, spoon is a
        hyponym of cutlery.

Ignore:
    class_energy.sum(dim=1)
    cond_probs.sum(dim=1)
    class_probs.sum(dim=1)

    import kwplot
    kwplot.autompl()
    import graphid
    graphid.util.show_nx(self.graph)

    dset = coco_dataset.CocoDataset.demo()
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import kwarray
import functools
import itertools as it
import networkx as nx
import ubelt as ub
import torch.nn.functional as F
import numpy as np


class CategoryTree(ub.NiceRepr):
    """
    Helps compute softmaxes and probabilities for tree-based categories
    where a directed edge (A, B) represents that A is a superclass of B.

    TODO:
        - [ ] separate the category tree from the method used to
              compute the hierarchical softmax

        - [ ] I think its ok to keep the decision function here.

        - [ ] Should we not use an implicit root by default?

    Example:
        >>> from ndsampler.category_tree import *
        >>> graph = nx.from_dict_of_lists({
        >>>     'background': [],
        >>>     'foreground': ['animal'],
        >>>     'animal': ['mammal', 'fish', 'insect', 'reptile'],
        >>>     'mammal': ['dog', 'cat', 'human', 'zebra'],
        >>>     'zebra': ['grevys', 'plains'],
        >>>     'grevys': ['fred'],
        >>>     'dog': ['boxer', 'beagle', 'golden'],
        >>>     'cat': ['maine coon', 'persian', 'sphynx'],
        >>>     'reptile': ['bearded dragon', 't-rex'],
        >>> }, nx.DiGraph)
        >>> self = CategoryTree(graph)
        >>> class_energy = torch.randn(3, len(self))
        >>> class_probs = self.hierarchical_softmax(class_energy, dim=1)
        >>> print(self)
        <CategoryTree(nNodes=22, maxDepth=6, maxBreadth=4...)>
    """
    def __init__(self, graph=None):
        """
        Args:
            graph (nx.DiGraph):
                either the graph representing a category hierarchy
        """
        if graph is None:
            graph = nx.DiGraph()
        else:
            if len(graph) > 0:
                if not nx.is_directed_acyclic_graph(graph):
                    raise ValueError('The category graph must a DAG')
                if not nx.is_forest(graph):
                    raise ValueError('The category graph must be a forest')
            if not isinstance(graph, nx.Graph):
                raise TypeError('Input to CategoryTree must be a networkx graph not {}'.format(type(graph)))
        self.graph = graph  # :type: nx.Graph
        # Note: nodes are class names
        self.id_to_node = None
        self.node_to_id = None
        self.node_to_idx = None
        self.idx_to_node = None
        self.idx_groups = None

        self._build_index()

    def copy(self):
        new = self.__class__(self.graph.copy())
        return new

    @classmethod
    def from_mutex(cls, nodes):
        """
        Args:
            nodes (List[str]): or a list of class names (in which case they
                will all be assumed to be mutually exclusive)

        Example:
            >>> print(CategoryTree.from_mutex(['a', 'b', 'c']))
            <CategoryTree(nNodes=3, maxDepth=1, maxBreadth=3...)>
        """
        nodes = list(nodes)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        start = 0
        if 'background' in graph.nodes:
            # hack
            graph.nodes['background']['id'] = 0
            start = 1

        for i, node in enumerate(nodes, start=start):
            graph.nodes[node]['id'] = graph.nodes[node].get('id', i)

        return cls(graph)

    @classmethod
    def from_json(cls, state):
        self = cls()
        self.__setstate__(state)
        return self

    @classmethod
    def from_coco(cls, categories):
        """
        Create a CategoryTree object from coco categories

        Args:
            List[Dict]: list of coco-style categories
        """
        graph = nx.DiGraph()
        for cat in categories:
            graph.add_node(cat['name'], **cat)
            if 'supercategory' in cat:
                graph.add_edge(cat['supercategory'], cat['name'])
        self = cls(graph)
        return self

    @classmethod
    def coerce(cls, data):
        """
        Attempt to coerce data as a CategoryTree object.

        This is primarily useful for when the software stack depends on
        categories being represent

        This will work if the input data is a specially formatted json dict, a
        list of mutually exclusive classes, or if it is already a CategoryTree.
        Otherwise an error will be thrown.

        Args:
            data (object): a known representation of a category tree.

        Returns:
            CategoryTree: self

        Raises:
            TypeError - if the input format is unknown
        """
        if isinstance(data, int):
            # An integer specifies the number of classes.
            self = cls.from_mutex(
                ['class_{}'.format(i + 1) for i in range(data)])
        elif isinstance(data, dict):
            # A dictionary is assumed to be in a special json format
            self = cls.from_json(data)
        elif isinstance(data, list):
            # A list is assumed to be a list of class names
            self = cls.from_mutex(data)
        elif isinstance(data, nx.DiGraph):
            # A nx.DiGraph should represent the category tree
            self = cls(data)
        elif isinstance(data, cls):
            # If data is already a CategoryTree, do nothing and just return it
            self = data
        else:
            raise TypeError('Unknown type {}: {!r}'.format(type(data), data))
        return self

    @classmethod
    def cast(cls, data):
        import warnings
        warnings.warn('cast is deprecated use coerce')
        return cls.coerce(data)

    @ub.memoize_property
    def id_to_idx(self):
        """
        Example:
            >>> import ndsampler
            >>> self = ndsampler.CategoryTree.demo()
            >>> self.id_to_idx[1]
        """
        return _calldict({cid: self.node_to_idx[node]
                         for cid, node in self.id_to_node.items()})

    @ub.memoize_property
    def idx_to_id(self):
        """
        Example:
            >>> import ndsampler
            >>> self = ndsampler.CategoryTree.demo()
            >>> self.idx_to_id[0]
        """
        return [self.node_to_id[node]
                for node in self.idx_to_node]

    @ub.memoize_method
    def idx_to_ancestor_idxs(self, include_self=True):
        """
        Mapping from a class index to its ancestors

        Args:
            include_self (bool, default=True):
                if True includes each node as its own ancestor.
        """
        lut = {
            idx: set(ub.take(self.node_to_idx, nx.ancestors(self.graph, node)))
            for idx, node in enumerate(self.idx_to_node)
        }
        if include_self:
            for idx, idxs in lut.items():
                idxs.update({idx})
        return lut

    @ub.memoize_method
    def idx_to_descendants_idxs(self, include_self=False):
        """
        Mapping from a class index to its descendants (including itself)

        Args:
            include_self (bool, default=False):
                if True includes each node as its own descendant.
        """
        lut = {
            idx: set(ub.take(self.node_to_idx, nx.descendants(self.graph, node)))
            for idx, node in enumerate(self.idx_to_node)
        }
        if include_self:
            for idx, idxs in lut.items():
                idxs.update({idx})
        return lut

    @ub.memoize_method
    def idx_pairwise_distance(self):
        """
        Get a matrix encoding the distance from one class to another.

        Distances
            * from parents to children are positive (descendants),
            * from children to parents are negative (ancestors),
            * between unreachable nodes (wrt to forward and reverse graph) are
            nan.
        """
        pdist = np.full((len(self), len(self)), fill_value=-np.nan,
                        dtype=np.float32)
        for node1, dists in nx.all_pairs_shortest_path_length(self.graph):
            idx1 = self.node_to_idx[node1]
            for node2, dist in dists.items():
                idx2 = self.node_to_idx[node2]
                pdist[idx1, idx2] = dist
                pdist[idx2, idx1] = -dist
        return pdist

    def __len__(self):
        return len(self.graph)

    def __iter__(self):
        return iter(self.idx_to_node)

    def __getitem__(self, index):
        return self.idx_to_node[index]

    def __json__(self):
        """
        Example:
            >>> import pickle
            >>> self = CategoryTree.demo()
            >>> print('self = {!r}'.format(self.__json__()))
        """
        return self.__getstate__()

    def __getstate__(self):
        """
        Serializes information in this class

        Example:
            >>> from ndsampler.category_tree import *
            >>> import pickle
            >>> self = CategoryTree.demo()
            >>> state = self.__getstate__()
            >>> serialization = pickle.dumps(self)
            >>> recon = pickle.loads(serialization)
            >>> assert recon.__json__() == self.__json__()
        """
        state = self.__dict__.copy()
        for key in list(state.keys()):
            if key.startswith('_cache'):
                state.pop(key)
        state['graph'] = to_directed_nested_tuples(self.graph)
        if True:
            # Remove reundant items
            state.pop('node_to_idx')
            state.pop('node_to_id')
            state.pop('idx_groups')
        return state

    def __setstate__(self, state):
        graph = from_directed_nested_tuples(state['graph'])

        if True:
            # Reconstruct redundant items
            if 'node_to_idx' not in state:
                state['node_to_idx'] = {node: idx for idx, node in
                                        enumerate(state['idx_to_node'])}
            if 'node_to_id' not in state:
                state['node_to_id'] = {node: id for id, node in
                                       state['id_to_node'].items()}

            if 'idx_groups' not in state:
                node_groups = list(traverse_siblings(graph))
                node_to_idx = state['node_to_idx']
                state['idx_groups'] = [sorted([node_to_idx[n] for n in group])
                                       for group in node_groups]

        self.__dict__.update(state)
        self.graph = graph

    def __nice__(self):
        return 'nNodes={}, maxDepth={}, maxBreadth={}, nodes={}'.format(
            self.num_classes,
            tree_depth(self.graph),
            max(it.chain([0], map(len, self.idx_groups))),
            self.idx_to_node,
        )

    def _demo_probs(self, num=5, rng=0, nonrandom=3, hackargmax=True):
        """ dummy probabilities for testing """
        rng = kwarray.ensure_rng(rng)
        class_energy = torch.FloatTensor(rng.rand(num, len(self)))

        # Setup the first few examples to prefer being classified
        # as a fine grained class to a decreasing degree.
        # The first example is set to have equal energy
        # The i + 2-th example is set to have an extremely high energy.
        start = 0
        nonrandom = min(nonrandom, (num - start))
        if nonrandom > 0:
            path = sorted(ub.take(self.node_to_idx, nx.dag_longest_path(self.graph)))

            class_energy[start] = 1 / len(class_energy[start])
            if hackargmax:
                # HACK: even though we want to test uniform distributions, it makes
                # regression tests difficiult because torch and numpy return a
                # different argmax when the array has more than one max value.
                # add a VERY small epsilon to make max values distinct
                class_energy[start] += torch.linspace(0, .00001, len(class_energy[start]))

            if nonrandom > 1:
                for i in range(nonrandom - 2):
                    class_energy[start + i + 1][path] += 2 ** (i / 4)
                class_energy[start + i + 2][path] += 2 ** 20

        class_probs = self.hierarchical_softmax(class_energy, dim=1)
        return class_probs

    @classmethod
    def demo(cls, key='coco', **kwargs):
        """
        Args:
            key (str): specify which demo dataset to use.

        CommandLine:
            xdoctest -m ~/code/ndsampler/ndsampler/category_tree.py CategoryTree.demo

        Example:
            >>> from ndsampler.category_tree import *
            >>> self = CategoryTree.demo()
            >>> print('self = {}'.format(self))
            self = <CategoryTree(nNodes=10, maxDepth=2, maxBreadth=4...)>
        """
        if key == 'coco':
            from ndsampler import coco_dataset
            dset = coco_dataset.CocoDataset.demo(**kwargs)
            dset.add_category('background', id=0)
            graph = dset.category_graph()
        elif key == 'btree':
            r = kwargs.pop('r', 3)
            h = kwargs.pop('h', 3)
            graph = nx.generators.balanced_tree(r=r, h=h, create_using=nx.DiGraph())
            graph = nx.relabel_nodes(graph, {n: n + 1 for n in graph})
            if kwargs.pop('add_zero', True):
                graph.add_node(0)
            assert not kwargs
        else:
            raise KeyError(key)
        self = cls(graph)
        return self

    def to_coco(self):
        """
        Converts to a coco-style data structure
        """
        for cid, node in self.id_to_node.items():
            # Skip if background already added
            cat = {
                'id': cid,
                'name': node,
            }
            parents = list(self.graph.predecessors(node))
            if len(parents) == 1:
                cat['supercategory'] = parents[0]
            else:
                if len(parents) > 1:
                    raise Exception('not a tree')
            yield cat

    def is_mutex(self):
        """
        Returns True if all categories are mutually exclusive (i.e. flat)

        If true, then the classes may be represented as a simple list of class
        names without any loss of information, otherwise the underlying
        category graph is necessary to preserve all knowledge.

        TODO:
            - [ ] what happens when we have a dummy root?
        """
        return len(self.graph.edges) == 0

    @property
    def num_classes(self):
        return self.graph.number_of_nodes()

    @property
    def class_names(self):
        return self.idx_to_node

    @property
    def category_names(self):
        return self.idx_to_node

    @property
    def cats(self):
        """
        Returns a mapping from category names to category attributes.

        If this category tree was constructed from a coco-dataset, then this
        will contain the coco category attributes.

        Returns:
            Dict[str, Dict[str, object]]

        Example:
            >>> from ndsampler.category_tree import *
            >>> self = CategoryTree.demo()
            >>> print('self.cats = {!r}'.format(self.cats))
        """
        return dict(self.graph.nodes)

    def index(self, node):
        """ Return the index that corresponds to the category name """
        return self.node_to_idx[node]

    def _build_index(self):
        """ construct lookup tables """
        # Most of the categories should have been given integer ids
        max_id = max(it.chain([0], nx.get_node_attributes(self.graph, 'id').values()))
        # Fill in id-values for any node that doesn't have one
        node_to_id = {}
        for node, attrs in sorted(self.graph.nodes.items()):
            node_to_id[node] = attrs.get('id', max_id + 1)
            max_id = max(max_id, node_to_id[node])
        id_to_node = ub.invert_dict(node_to_id)

        # Compress ids into a flat index space (sorted by node ids)
        idx_to_node = ub.argsort(node_to_id)
        node_to_idx = {node: idx for idx, node in enumerate(idx_to_node)}

        # Find the sets of nodes that need to be softmax-ed together
        node_groups = list(traverse_siblings(self.graph))
        idx_groups = [sorted([node_to_idx[n] for n in group])
                      for group in node_groups]

        # Set instance attributes
        self.id_to_node = id_to_node
        self.node_to_id = node_to_id
        self.idx_to_node = idx_to_node
        self.node_to_idx = node_to_idx
        self.idx_groups = idx_groups

    def conditional_log_softmax(self, class_energy, dim):
        """
        Computes conditional log probabilities of each class in the category tree

        Args:
            class_energy (Tensor): raw values output by final network layer
            dim (int): dimension where each index corresponds to a class

        Example:
            >>> from ndsampler.category_tree import *
            >>> graph = nx.generators.gnr_graph(30, 0.3, seed=321).reverse()
            >>> self = CategoryTree(graph)
            >>> class_energy = torch.randn(64, len(self.idx_to_node))
            >>> cond_logprobs = self.conditional_log_softmax(class_energy, dim=1)
            >>> # The sum of the conditional probabilities should be
            >>> # equal to the number of sibling groups.
            >>> cond_probs = torch.exp(cond_logprobs).numpy()
            >>> assert np.allclose(cond_probs.sum(axis=1), len(self.idx_groups))
        """
        cond_logprobs = torch.empty_like(class_energy)
        if class_energy.numel() == 0:
            return cond_logprobs
        # Move indexes onto the class_energy device (perhaps precache this)
        index_groups = [torch.LongTensor(idxs).to(class_energy.device)
                        for idxs in self.idx_groups]
        # Note: benchmarks for implementation choices are in ndsampler/dev
        # The index_select/index_copy_ solution is faster than fancy indexing
        for index in index_groups:
            # Take each subset of classes that are mutually exclusive
            energy_group = torch.index_select(class_energy, dim=dim, index=index)
            # Then apply the log_softmax to those sets
            logprob_group = F.log_softmax(energy_group, dim=dim)
            cond_logprobs.index_copy_(dim, index, logprob_group)
        return cond_logprobs

    def _apply_logit_chain_rule(self, cond_logprobs, dim):
        """
        Incorrect name for backwards compatibility
        """
        import warnings
        warnings.warn(
            'Use _apply_logit_chain_rule is incorrectly named '
            'and deprecated use _apply_logprob_chain_rule instead',
            DeprecationWarning)
        return self._apply_logprob_chain_rule(cond_logprobs, dim)

    def _apply_logprob_chain_rule(self, cond_logprobs, dim):
        """
        Applies the probability chain rule (in log space, which has better
        numerical properties) to a set of conditional probabilities (wrt this
        hierarchy) to achieve absolute probabilities for each node.

        Args:
            cond_logprobs (Tensor): conditional log probabilities for each class
            dim (int): dimension where each index corresponds to a class

        Notes:
            Probability chain rule:
                P(node) = P(node | parent) * P(parent)

            Log-Probability chain rule:
                log(P(node)) = log(P(node | parent)) + log(P(parent))
        """
        # The dynamic program was faster on the CPU in a dummy test case
        memo = {}

        def log_prob(node, memo=memo):
            """ dynamic program to compute absolute class log probability """
            if node in memo:
                return memo[node]
            logp_node_given_parent = cond_logprobs.select(dim, self.node_to_idx[node])
            parents = list(self.graph.predecessors(node))
            if len(parents) == 0:
                logp_node = logp_node_given_parent
            elif len(parents) == 1:
                # log(P(node)) = log(P(node | parent)) + log(P(parent))
                logp_node = logp_node_given_parent + log_prob(parents[0])
            else:
                raise AssertionError('not a tree')
            memo[node] = logp_node
            return logp_node

        class_logprobs = torch.empty_like(cond_logprobs)
        if cond_logprobs.numel() > 0:
            for idx, node in enumerate(self.idx_to_node):
                # Note: the this is the bottleneck in this function
                if True:
                    class_logprobs.select(dim, idx)[:] = log_prob(node)
                else:
                    result = log_prob(node)  # 50% of the time
                    _dest = class_logprobs.select(dim, idx)  # 8% of the time
                    _dest[:] = result  # 37% of the time
        return class_logprobs

    def source_log_softmax(self, class_energy, dim):
        """
        Top-down hierarchical softmax

        Alternative to `sink_log_softmax`

        SeeAlso:
            * sink_log_softmax
            * hierarchical_log_softmax

        Converts raw class energy to absolute log probabilites based on the
        category hierarchy. This is done by first converting to conditional
        log probabilities and then applying the probability chain rule (in log
        space, which has better numerical properties).

        Args:
            class_energy (Tensor): raw output from network. The values in
                `class_energy[..., idx]` should correspond to the network
                activations for the hierarchical class `self.idx_to_node[idx]`
            dim (int): dimension corresponding to classes (usually 1)

        Example:
            >>> from ndsampler.category_tree import *
            >>> graph = nx.generators.gnr_graph(20, 0.3, seed=328).reverse()
            >>> self = CategoryTree(graph)
            >>> class_energy = torch.randn(3, len(self.idx_to_node))
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> class_probs = torch.exp(class_logprobs)
            >>> for node, idx in self.node_to_idx.items():
            ...     # Check the total children probabilities
            ...     # is equal to the parent probablity.
            ...     children = list(self.graph.successors(node))
            ...     child_idxs = [self.node_to_idx[c] for c in children]
            ...     if len(child_idxs) > 0:
            ...         child_sum = class_probs[..., child_idxs].sum(dim=1)
            ...         p_node = class_probs[..., idx]
            ...         torch.allclose(child_sum, p_node)
        """
        if dim < 0:
            dim = dim % class_energy.ndimension()
        cond_logprobs = self.conditional_log_softmax(class_energy, dim=dim)
        class_logprobs = self._apply_logprob_chain_rule(cond_logprobs, dim=dim)
        return class_logprobs

    def sink_log_softmax(self, class_energy, dim):
        """
        Bottom-up hierarchical softmax

        Alternative to `source_log_softmax`

        SeeAlso:
            * source_log_softmax
            * hierarchical_log_softmax

        Does a regular softmax over all the mutually exclusive leaf nodes, then
        sums their probabilities to get the score for the parent nodes.

        Args:
            class_energy (Tensor): raw output from network. The values in
                `class_energy[..., idx]` should correspond to the network
                activations for the hierarchical class `self.idx_to_node[idx]`
            dim (int): dimension corresponding to classes (usually 1)

        Example:
            >>> from ndsampler.category_tree import *
            >>> graph = nx.generators.gnr_graph(20, 0.3, seed=328).reverse()
            >>> self = CategoryTree(graph)
            >>> class_energy = torch.randn(3, len(self.idx_to_node))
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=1)
            >>> class_probs = torch.exp(class_logprobs)
            >>> for node, idx in self.node_to_idx.items():
            ...     # Check the total children probabilities
            ...     # is equal to the parent probablity.
            ...     children = list(self.graph.successors(node))
            ...     child_idxs = [self.node_to_idx[c] for c in children]
            ...     if len(child_idxs) > 0:
            ...         child_sum = class_probs[..., child_idxs].sum(dim=1)
            ...         p_node = class_probs[..., idx]
            ...         torch.allclose(child_sum, p_node)

        Ignore:
            >>> class_logprobs1 = self.sink_log_softmax(class_energy, dim=1)
            >>> class_logprobs2 = self.source_log_softmax(class_energy, dim=1)
            >>> class_probs1 = torch.exp(class_logprobs1)
            >>> class_probs2 = torch.exp(class_logprobs2)
        """
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
                total = torch.logsumexp(selected, dim=dim)  # sum in logspace
                class_logprobs.select(dim, node_idx)[:] = total

        for node in self.graph.nodes():
            populate2(node)
        return class_logprobs

    # TODO: Need to figure out the best way to parametarize which
    # of the source or sink softmaxes will be used by a network
    hierarchical_log_softmax = source_log_softmax
    # hierarchical_log_softmax = sink_log_softmax

    def hierarchical_softmax(self, class_energy, dim):
        """ Convinience method which converts class-energy to final probs """
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim)
        class_probs = torch.exp(class_logprobs)
        return class_probs

    def graph_log_softmax(self, class_energy, dim):
        """ Convinience method which converts class-energy to logprobs """
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim)
        return class_logprobs

    def graph_softmax(self, class_energy, dim):
        """ Convinience method which converts class-energy to final probs """
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim)
        class_probs = torch.exp(class_logprobs)
        return class_probs

    def hierarchical_cross_entropy(self, class_energy, targets,
                                   reduction='mean'):
        """
        Combines hierarchical_log_softmax and nll_loss in a single function
        """
        class_logprobs = self.hierarchical_log_softmax(class_energy, dim=1)
        loss = F.nll_loss(class_logprobs, targets, reduction=reduction)
        return loss

    def hierarchical_nll_loss(self, class_logprobs, targets):
        """
        Given predicted hierarchical class log-probabilities and target vectors
        indicating the finest-grained known target class, compute the loss such
        that only errors on coarser levels of the hierarchy are considered.
        To quote from YOLO-9000:
            > For example, if the label is “dog” we do assign any error to
            > predictions further down in the tree, “German Shepherd” versus
            > “Golden Retriever”, because we do not have that information.

        Note that all the hard word needed to consider all coarser classes and
        ignore all finer grained classes is done in the computation of
        `class_logprobs`. Given these and targets specified at the finest known
        category for each target (which might be very coarse), we can simply
        use regular nll_loss and everything will work out.

        This is because the class logprobs have already been computed using
        information from all conditional probabilities computed at itself and
        at each ancestor in the tree, and these conditional probabilities were
        computed with respect to all siblings at a given level, so the class
        logprobs exactly contain the relevant information. In other words as long
        as the log probabilities are computed correctly, then the nothing
        special needs to happen when computing the loss.

        Args:
            class_logprobs (Tensor): log probabilities for each class
            targets (Tensor): true class for each example

        Example:
            >>> from ndsampler.category_tree import *
            >>> graph = nx.from_dict_of_lists({
            >>>     'background': [],
            >>>     'mineral': ['granite', 'quartz'],
            >>>     'animal': ['dog', 'cat'],
            >>>     'dog': ['boxer', 'beagle']
            >>> }, nx.DiGraph)
            >>> self = CategoryTree(graph)
            >>> target_nodes = ['boxer', 'dog', 'quartz', 'animal', 'background']
            >>> targets = torch.LongTensor([self.node_to_idx[n] for n in target_nodes])
            >>> class_energy = torch.randn(len(targets), len(self), requires_grad=True)
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> loss = self.hierarchical_nll_loss(class_logprobs, targets)
            >>> # Note that only the relevant classes (wrt to the target) for
            >>> # each batch item receive gradients
            >>> loss.backward()
            >>> is_relevant = (class_energy.grad.abs() > 0).numpy()
            >>> print('target -> relevant')
            >>> print('------------------')
            >>> for bx, idx in enumerate(targets.tolist()):
            ...     target_node = self.idx_to_node[idx]
            ...     relevant_nodes = list(ub.take(self.idx_to_node, np.where(is_relevant[bx])[0]))
            ...     print('{} -> {}'.format(target_node, relevant_nodes))
            target -> relevant
            ------------------
            boxer -> ['animal', 'background', 'beagle', 'boxer', 'cat', 'dog', 'mineral']
            dog -> ['animal', 'background', 'cat', 'dog', 'mineral']
            quartz -> ['animal', 'background', 'granite', 'mineral', 'quartz']
            animal -> ['animal', 'background', 'mineral']
            background -> ['animal', 'background', 'mineral']
        """
        loss = F.nll_loss(class_logprobs, targets)
        return loss

    def show(self):
        """
        Ignore:
            >>> import kwplot
            >>> kwplot.autompl()
            >>> from ndsampler import category_tree
            >>> self = category_tree.CategoryTree.demo()
            >>> self.show()

            python -c "import kwplot, ndsampler, graphid; kwplot.autompl(); graphid.util.show_nx(ndsampler.category_tree.CategoryTree.demo().graph); kwplot.show_if_requested()" --show
        """
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
        except ImportError:
            import warnings
            warnings.warn('pygraphviz is not available')
            pos = None
        nx.draw_networkx(self.graph, pos=pos)
        # import graphid
        # graphid.util.show_nx(self.graph)

    def _prob_decision(self, class_probs, dim, thresh=0.1):
        """
        Chooses the finest-grained category based on raw prob threshold

        Args:
            thresh (float): only make a more fine-grained decision if
                its probability is above this threshold. This number should
                be set relatively low to encourage smoothness in the
                detectio metrics.

        Example:
            >>> from ndsampler.category_tree import *
            >>> import torch
            >>> from ndsampler import category_tree
            >>> self = category_tree.CategoryTree.demo('btree', r=3, h=4)
            >>> class_energy = torch.randn(33, len(self))
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> class_probs = torch.exp(class_logprobs).numpy()
            >>> dim = 1
            >>> thresh = 0.3
            >>> pred_idxs, pred_conf = self.decision(class_probs, dim, thresh=thresh)
            >>> pred_cnames = list(ub.take(self.idx_to_node, pred_idxs))
        """
        import kwarray
        impl = kwarray.ArrayAPI.impl(class_probs)

        sources = list(source_nodes(self.graph))
        other_dims = sorted(set(range(len(class_probs.shape))) - {dim})

        # Rearange probs so the class dimension is at the end
        # flat_class_probs = class_probs.transpose(*other_dims + [dim]).reshape(-1, class_probs.shape[dim])
        flat_class_probs = impl.transpose(class_probs, other_dims + [dim]).reshape(-1, class_probs.shape[dim])
        flat_jdxs = np.arange(flat_class_probs.shape[0])

        def _descend(depth, nodes, jdxs):
            """
            Recursively descend the class tree starting at the coursest level.
            At each level we decide if the items will take a category at this
            level of granulatority or try to take a more fine-grained label.

            Args:
                depth (int): current depth in the tree
                nodes (list) : set of sibling nodes at a this level
                jdxs (ArrayLike): item indices that made it to this level (note
                    idxs are used for class indices)
            """
            # Look at the probabilities of each node at this level
            idxs = sorted(self.node_to_idx[node] for node in nodes)
            probs = flat_class_probs[jdxs][:, idxs]

            pred_conf, pred_cx = impl.max_argmax(probs, axis=1)
            pred_idxs = np.array(idxs)[pred_cx]

            # Keep desending on items above the threshold
            # TODO: is there a more intelligent way to do this?
            check_children = pred_conf > thresh

            if impl.any(check_children):
                # Check the children of these nodes
                check_jdxs = jdxs[check_children]
                check_idxs = pred_idxs[check_children]
                group_idxs, groupxs = kwarray.group_indices(check_idxs)
                for idx, groupx in zip(group_idxs, groupxs):
                    node = self.idx_to_node[idx]
                    children = list(self.graph.successors(node))
                    if children:
                        sub_jdxs = check_jdxs[groupx]
                        # See if any fine-grained categories also have high
                        # thresholds.
                        sub_idxs, sub_conf = _descend(depth + 1, children,
                                                      sub_jdxs)
                        sub_flags = sub_conf > thresh
                        # Overwrite course decisions with confident
                        # fine-grained ones.
                        fine_groupx = groupx[sub_flags]
                        fine_idxs = sub_idxs[sub_flags]
                        fine_conf = sub_conf[sub_flags]
                        pred_conf[fine_groupx] = fine_conf
                        pred_idxs[fine_groupx] = fine_idxs
            return pred_idxs, pred_conf

        nodes = sources
        jdxs = flat_jdxs
        pred_idxs, pred_conf = _descend(0, nodes, jdxs)
        return pred_idxs, pred_conf

    def decision(self, class_probs, dim, thresh=0.5, criterion='gini',
                 ignore_class_idxs=None, always_refine_idxs=None):
        """
        Chooses the finest-grained category based on information gain

        Args:
            thresh (float): threshold on simplicity ratio.
                Small thresholds are more permissive, i.e. the returned classes
                will often be more fined-grained. Larger thresholds are less
                permissive and prefer coarse-grained classes.

            criterion (str): how to compute information. Either entropy or gini.

            ignore_class_idxs (List[int], optional): if specified this is a list
                of class indices which we are not allowed to predict. We
                will procede as if the graph did not contain these nodes.
                (Useful for getting low-probability detections).

            always_refine_idxs  (List[int], optional):
                if specified this is a list of class indices that we will
                always refine into a more fine-grained class.
                (Useful if you have a dummy root)

        Returns:
            Tuple[Tensor, Tensor]: pred_idxs, pred_conf:
                pred_idxs: predicted class indices
                pred_conf: associated confidence

        Example:
            >>> from ndsampler.category_tree import *
            >>> self = CategoryTree.demo('btree', r=3, h=3)
            >>> rng = kwarray.ensure_rng(0)
            >>> class_energy = torch.FloatTensor(rng.rand(33, len(self)))
            >>> # Setup the first few examples to prefer being classified
            >>> # as a fine grained class to a decreasing degree.
            >>> # The first example is set to have equal energy
            >>> # The i + 2-th example is set to have an extremely high energy.
            >>> path = sorted(ub.take(self.node_to_idx, nx.dag_longest_path(self.graph)))
            >>> class_energy[0] = 1 / len(class_energy[0])
            >>> for i in range(8):
            >>>     class_energy[i + 1][path] += 2 ** (i / 4)
            >>> class_energy[i + 2][path] += 2 ** 20
            >>> class_logprobs = self.hierarchical_log_softmax(class_energy, dim=-1)
            >>> class_probs = torch.exp(class_logprobs).numpy()
            >>> print(ub.hzcat(['probs = ', ub.repr2(class_probs[:8], precision=2, supress_small=True)]))
            >>> dim = 1
            >>> criterion = 'entropy'
            >>> thresh = 0.40
            >>> pred_idxs, pred_conf = self.decision(class_probs[0:10], dim, thresh=thresh, criterion=criterion)
            >>> print('pred_conf = {!r}'.format(pred_conf))
            >>> print('pred_idxs = {!r}'.format(pred_idxs))
            >>> pred_cnames = list(ub.take(self.idx_to_node, pred_idxs))

        Example:
            >>> from ndsampler.category_tree import *
            >>> self = CategoryTree.demo('btree', r=3, h=3)
            >>> class_probs = self._demo_probs()
            >>> # Test ignore_class_idxs
            >>> self.decision(class_probs, dim=1, ignore_class_idxs=[0])
            >>> # Should not be able to ignore all top level nodes
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     self.decision(class_probs, dim=1, ignore_class_idxs=self.idx_groups[0])
            >>> # But it is OK to ignore all child nodes at a particular level
            >>> self.decision(class_probs, dim=1, ignore_class_idxs=self.idx_groups[1])

        Example:
            >>> from ndsampler.category_tree import *
            >>> self = CategoryTree.demo('btree', r=3, h=3, add_zero=False)
            >>> class_probs = self._demo_probs(num=30, nonrandom=20)
            >>> pred_idxs0, pref_conf0 = self.decision(class_probs, dim=1, always_refine_idxs=[])
            >>> assert 0 in pred_idxs0
            >>> ###
            >>> #print(ub.color_text('!!!!!!!!!!!!!!!!!!!', 'white'))
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1, always_refine_idxs=[0])
            >>> #print(ub.color_text('!!!!!!!!!!!!!!!!!!!', 'red'))
            >>> pred_idxs2, pref_conf2 = self.decision(class_probs.numpy(), dim=1, always_refine_idxs=[0])
            >>> assert np.all(pred_idxs1 == pred_idxs2)
            >>> assert 0 not in pred_idxs1

        Example:
            >>> from ndsampler.category_tree import *
            >>> graph = nx.from_dict_of_lists({
            >>>     'a': ['b', 'q'],
            >>>     'b': ['c'],
            >>>     'c': ['d'],
            >>>     'd': ['e', 'f'],
            >>> }, nx.DiGraph)
            >>> self = CategoryTree(graph)
            >>> class_probs = self._demo_probs()
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1)
            >>> self = CategoryTree.demo('btree', r=1, h=4, add_zero=False)
            >>> class_probs = self._demo_probs()
            >>> # We should always descend to the finest level if we just have a straight line
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1)
            >>> assert np.all(pred_idxs1 == 4)

        Example:
            >>> # FIXME: What do we do in this case?
            >>> # Do we always decend at level A?
            >>> from ndsampler.category_tree import *
            >>> graph = nx.from_dict_of_lists({
            >>>     'a': ['b', 'c'],
            >>> }, nx.DiGraph)
            >>> self = CategoryTree(graph)
            >>> class_probs = self._demo_probs(num=10, nonrandom=8)
            >>> pred_idxs1, pref_conf1 = self.decision(class_probs, dim=1)
            >>> print('pred_idxs1 = {!r}'.format(pred_idxs1))
        """
        if criterion == 'prob':
            return self._prob_decision(class_probs, dim, thresh=thresh)

        DEBUG = False

        impl = kwarray.ArrayAPI.impl(class_probs)

        sources = list(source_nodes(self.graph))
        other_dims = sorted(set(range(len(class_probs.shape))) - {dim})

        # Rearange probs so the class dimension is at the end
        flat_class_probs = impl.transpose(class_probs, other_dims + [dim]).reshape(-1, class_probs.shape[dim])
        flat_jdxs = np.arange(flat_class_probs.shape[0])

        if criterion == 'gini':
            _criterion = functools.partial(gini, axis=dim, impl=impl)
            # _criterion = functools.partial(gini, axis=dim)
        elif criterion == 'entropy':
            _criterion = functools.partial(entropy, axis=dim, impl=impl)
            # _criterion = functools.partial(entropy, axis=dim)
        else:
            raise KeyError(criterion)

        def _entropy_refine(depth, nodes, jdxs):
            """
            Recursively descend the class tree starting at the coursest level.
            At each level we decide if the items will take a category at this
            level of granulatority or try to take a more fine-grained label.

            Args:
                depth (int): current depth in the tree
                nodes (list) : set of sibling nodes at a this level
                jdxs (ArrayLike): item indices that made it to this level (note
                    idxs are used for class indices)
            """
            if DEBUG:
                print(ub.color_text('* REFINE nodes={}'.format(nodes), 'blue'))
            # Look at the probabilities of each node at this level
            idxs = sorted(self.node_to_idx[node] for node in nodes)
            if ignore_class_idxs:
                ignore_nodes = set(ub.take(self.idx_to_node, ignore_class_idxs))
                idxs = sorted(set(idxs) - set(ignore_class_idxs))
                if len(idxs) == 0:
                    raise ValueError('Cannot ignore all top-level classes')
            probs = flat_class_probs[jdxs][:, idxs]

            # Choose a highest probability category to predict at this level
            pred_conf, pred_cx = impl.max_argmax(probs, axis=1)
            pred_idxs = np.array(idxs)[impl.numpy(pred_cx)]

            # Group each example which predicted the same class at this level
            group_idxs, groupxs = kwarray.group_indices(pred_idxs)
            if DEBUG:
                groupxs = list(ub.take(groupxs, group_idxs.argsort()))
                group_idxs = group_idxs[group_idxs.argsort()]
                # print('groupxs = {!r}'.format(groupxs))
                # print('group_idxs = {!r}'.format(group_idxs))

            for idx, groupx in zip(group_idxs, groupxs):
                # Get the children of this node (idx)
                node = self.idx_to_node[idx]
                children = sorted(self.graph.successors(node))
                if ignore_class_idxs:
                    children = sorted(set(children) - ignore_nodes)

                if children:
                    # Check if it would be simple to refine the coarse category
                    # current prediction into one of its finer-grained child
                    # categories. Do this by considering the entropy at this
                    # level if we replace this coarse-node with the child
                    # fine-nodes. Then compare that entropy to what we would
                    # get if we were perfectly uncertain about the child node
                    # prediction (i.e. the worst case). If the entropy we get
                    # is much lower than the worst case, then it is simple to
                    # descend the tree and predict a finer-grained label.

                    # Expand this node into all of its children
                    child_idxs = set(self.node_to_idx[child] for child in children)

                    # Get example indices (jdxs) assigned to category idx
                    groupx.sort()
                    group_jdxs = jdxs[groupx]

                    # Expand this parent node, but keep the parent's siblings
                    ommer_idxs = sorted(set(idxs) - {idx})  # Note: ommer = Aunt/Uncle
                    expanded_idxs = sorted(ommer_idxs) + sorted(child_idxs)
                    expanded_probs = flat_class_probs[group_jdxs][:, expanded_idxs]

                    # Compute the entropy of the expanded distribution
                    h_expanded = _criterion(expanded_probs)

                    # Probability assigned to the parent
                    p_parent = flat_class_probs[group_jdxs][:, idx:idx + 1]
                    # Get the absolute probabilities assigned the parents siblings
                    ommer_probs = flat_class_probs[group_jdxs][:, sorted(ommer_idxs)]

                    # Compute the worst-case entropy after expanding the node
                    # In the worst case the parent probability is distributed
                    # uniformly among all of its children
                    c = len(children)
                    child_probs_worst = impl.tile(p_parent / c, reps=[1, c])
                    expanded_probs_worst = impl.hstack([ommer_probs, child_probs_worst])
                    h_expanded_worst = _criterion(expanded_probs_worst)

                    # Normalize the entropy we got by the worst case.
                    # eps = float(np.finfo(np.float32).min)
                    eps = 1e-30
                    complexity_ratio = h_expanded / (h_expanded_worst + eps)
                    simplicity_ratio = 1 - complexity_ratio

                    # If simplicity ratio is over a threshold refine the parent
                    refine_flags = simplicity_ratio > thresh

                    if always_refine_idxs is not None:
                        if idx in always_refine_idxs:
                            refine_flags[:] = 1

                    refine_flags = kwarray.ArrayAPI.numpy(refine_flags).astype(np.bool)

                    if DEBUG:
                        print('-----------')
                        print('idx = {!r}'.format(idx))
                        print('ommer_idxs = {!r}'.format(ommer_idxs))
                        print('depth = {!r}'.format(depth))
                        import pandas as pd
                        print('expanded_probs =\n{}'.format(
                            ub.repr2(expanded_probs, precision=2, supress_small=True)))
                        df = pd.DataFrame({
                            'h': h_expanded,
                            'h_worst': h_expanded_worst,
                            'ratio': complexity_ratio,
                            'flags': refine_flags.astype(np.uint8)
                        })
                        print(df)
                        print('-----------')

                    if np.any(refine_flags):
                        refine_jdxs = group_jdxs[refine_flags]
                        refine_idxs, refine_conf = _entropy_refine(depth + 1, children, refine_jdxs)
                        # Overwrite course decisions with refined decisions.
                        refine_groupx = groupx[refine_flags]
                        pred_idxs[refine_groupx] = refine_idxs
                        pred_conf[refine_groupx] = refine_conf
            return pred_idxs, pred_conf

        nodes = sources
        jdxs = flat_jdxs
        depth = 0
        pred_idxs, pred_conf = _entropy_refine(depth, nodes, jdxs)
        return pred_idxs, pred_conf

    #### DEPRECATED METHODS

    def heirarchical_softmax(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_softmax(*args, **kw)

    def heirarchical_cross_entropy(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_cross_entropy(*args, **kw)

    def heirarchical_nll_loss(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_nll_loss(*args, **kw)

    def heirarchical_log_softmax(self, *args, **kw):
        import warnings
        warnings.warn('deprecated use the correctly spelled version')
        return self.hierarchical_log_softmax(*args, **kw)


def source_nodes(graph):
    """ generates source nodes --- nodes without incoming edges """
    return (n for n in graph.nodes() if graph.in_degree(n) == 0)


def sink_nodes(graph):
    """ generates source nodes --- nodes without incoming edges """
    return (n for n in graph.nodes() if graph.out_degree(n) == 0)


def traverse_siblings(graph, sources=None):
    """ generates groups of nodes that have the same parent """
    if sources is None:
        sources = list(source_nodes(graph))
    yield sources
    for node in sources:
        children = list(graph.successors(node))
        if children:
            for _ in traverse_siblings(graph, children):
                yield _


def tree_depth(graph, root=None):
    """
    Maximum depth of the forest / tree

    Example:
        >>> from ndsampler.category_tree import *
        >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        >>> tree_depth(graph)
        4
        >>> tree_depth(nx.balanced_tree(r=2, h=0, create_using=nx.DiGraph))
        1
    """
    if len(graph) == 0:
        return 0
    if root is not None:
        assert root in graph.nodes
    assert nx.is_forest(graph)
    def _inner(root):
        if root is None:
            return max(it.chain([0], (_inner(n) for n in source_nodes(graph))))
        else:
            return max(it.chain([0], (_inner(n) for n in graph.successors(root)))) + 1
    depth = _inner(root)
    return depth


def to_directed_nested_tuples(graph, with_data=True):
    """
    Encodes each node and its children in a tuple as:
        (node, children)
    """
    def _represent_node(node):
        if with_data:
            node_data = graph.nodes[node]
            return (node, node_data, _traverse_encode(node))
        else:
            return (node, _traverse_encode(node))

    def _traverse_encode(parent):
        children = sorted(graph.successors(parent))
        # graph.get_edge_data(node, child)
        return [_represent_node(node) for node in children]

    sources = sorted(source_nodes(graph))
    encoding = [_represent_node(node) for node in sources]
    return encoding


def from_directed_nested_tuples(encoding):
    """
    Example:
        >>> from ndsampler.category_tree import *
        >>> graph = nx.generators.gnr_graph(20, 0.3, seed=790).reverse()
        >>> graph.nodes[0]['color'] = 'black'
        >>> encoding = to_directed_nested_tuples(graph)
        >>> recon = from_directed_nested_tuples(encoding)
        >>> recon_encoding = to_directed_nested_tuples(recon)
        >>> assert recon_encoding == encoding
    """
    node_data_view = {}
    def _traverse_recon(tree):
        nodes = []
        edges = []
        for tup in tree:
            if len(tup) == 2:
                node, subtree = tup
            elif len(tup) == 3:
                node, node_data, subtree = tup
                node_data_view[node] = node_data
            else:
                raise AssertionError('invalid tup')
            children = [t[0] for t in subtree]
            nodes.append(node)
            edges.extend((node, child) for child in children)
            subnodes, subedges = _traverse_recon(subtree)
            nodes.extend(subnodes)
            edges.extend(subedges)
        return nodes, edges
    nodes, edges = _traverse_recon(encoding)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    for k, v in node_data_view.items():
        graph.nodes[k].update(v)
    return graph


def gini(probs, axis=1, impl=np):
    """
    Approximates Shannon Entropy, but faster to compute

    Example:
        >>> rng = kwarray.ensure_rng(0)
        >>> probs = torch.softmax(torch.Tensor(rng.rand(3, 10)), 1)
        >>> gini(probs.numpy(), impl=kwarray.ArrayAPI.coerce('numpy'))
        array...0.896..., 0.890..., 0.892...
        >>> gini(probs, impl=kwarray.ArrayAPI.coerce('torch'))
        tensor...0.896..., 0.890..., 0.892...
    """
    return 1 - impl.sum(probs ** 2, axis=axis)


def entropy(probs, axis=1, impl=np):
    """
    Standard Shannon (Information Theory) Entropy

    Example:
        >>> rng = kwarray.ensure_rng(0)
        >>> probs = torch.softmax(torch.Tensor(rng.rand(3, 10)), 1)
        >>> entropy(probs.numpy(), impl=kwarray.ArrayAPI.coerce('numpy'))
        array...3.295..., 3.251..., 3.265...
        >>> entropy(probs, impl=kwarray.ArrayAPI.coerce('torch'))
        tensor...3.295..., 3.251..., 3.265...
    """
    with np.errstate(divide='ignore'):
        logprobs = impl.log2(probs)
        logprobs = impl.nan_to_num(logprobs, copy=False)
        h = -impl.sum(probs * logprobs, axis=axis)
        return h


class _calldict(dict):
    """
    helper object to maintain backwards compatibility between new and old
    id_to_idx methods.

    Example:
        >>> self = _calldict({1: 2})
        >>> #assert self()[1] == 2
        >>> assert self[1] == 2
    """

    def __call__(self):
        import warnings
        warnings.warn('Calling id_to_idx as a method has been depricated. '
                      'Use this dict as a property')
        return self


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m ndsampler.category_tree
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
