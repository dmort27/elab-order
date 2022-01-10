import graphviz, pydot
from IPython.display import Image, SVG
from collections import defaultdict
import re


class TreeTrimmer:
    '''
    Given a trained decision tree in the DOT format, extract the d nodes on the left most branch (the 'no' branch)
    along with their children in order to extract the linear scale learned by the decision tree
    '''
    def __init__(self, dot_file, max_depth):
        self.nodes = {}
        self.edges = defaultdict(list)
        self.max_depth = max_depth
        with open(dot_file) as f:
            for line in f:
                if line.startswith('}'): break
                t0, t1 = line.split(' ', maxsplit=1)
                if t0 in ['digraph', 'node', 'edge']:
                    continue
                if t1.startswith('['):  # a new node
                    self.nodes[t0] = t1
                elif t1.startswith('->'):  # a new edge
                    self.edges[t0].append(re.search(r'-> (.*?)[; ]', t1).group(1))

    def left_child(self, node):
        return self.edges[node][0]

    def right_child(self, node):
        return self.edges[node][1]

    def transform_splitter(self, old_splitter):
        m = re.search(r'cc([1|2])_(.*?)_(.*?) <= 0.5', old_splitter)
        word, constituent, phoneme = m.groups()
        varnothing = '∅'  # '\\varnothing'
        word = '₁' if word == '1' else '₂'
        return f"{constituent}(B{word}) = {phoneme or varnothing} ?"

    def transform_label(self, node, omit_splitter=False):
        old_label = re.search(r'label="(.*?)"', self.nodes[node]).group(1)
        lines = old_label.split('\\n')
        if len(lines) == 4:
            # normal: splitter, samples, value, class
            if omit_splitter:
                splitter = '(...)'
            else:
                splitter = self.transform_splitter(lines[0])
            out_lines = "{" + f"{lines[3]}\n{lines[2]}|{splitter}" + "}"
        elif len(lines) == 3:
            # terminal node: samples, value, class
            out_lines = f"{lines[2]}\n{lines[1]}"
        else:
            out_lines = '???'

        return out_lines

    def run(self, out_name, title=''):
        label_kwargs = {
            'label': '<<B>'+title+'</B>>',
            'labelloc': 't',
            'labelfontsize': 24,
        }
        graph = pydot.Dot("trimmed_tree", graph_type="digraph", **label_kwargs)
        graph.set_node_defaults(shape='record')

        root = pydot.Node("a0", label=self.transform_label('0'))
        graph.add_node(root)
        node_a = '0'
        for i, d in enumerate(range(self.max_depth)):
            z = pydot.Node(f"z{i + 1}", style='invisible')
            a = pydot.Node(f"a{i + 1}", label=self.transform_label(
                self.left_child(node_a), omit_splitter=(i==self.max_depth-1)))
            b = pydot.Node(f"b{i + 1}", label=self.transform_label(
                self.right_child(node_a), omit_splitter=True))
            graph.add_node(z)
            graph.add_node(a)
            graph.add_node(b)
            z_edge = pydot.Edge(f"a{i}", f"z{i + 1}", style="invisible", arrowhead="none")
            a_edge = pydot.Edge(f"a{i}", f"a{i + 1}", headlabel="No", labelangle=45, labeldistance=2.5)
            b_edge = pydot.Edge(f"a{i}", f"b{i + 1}", headlabel="Yes", labelangle=-45, labeldistance=2.5)
            graph.add_edge(z_edge)
            graph.add_edge(a_edge)
            graph.add_edge(b_edge)
            node_a = self.left_child(node_a)

        graph.write_pdf(out_name)


if __name__ == '__main__':
    trimmer = TreeTrimmer("../out/hmn-Latn_ton_paperver_0.dot", 9)
    trimmer.run("../out/hmong_tree.pdf", title='Hmong Tones')
    trimmer = TreeTrimmer("../out/lhu-Latn_rhy_paperver.dot", 9)
    trimmer.run("../out/lahu_tree.pdf", title='Lahu Rhymes')
    trimmer = TreeTrimmer("../out/ltc-IPA_ton_paperver.dot", 9)
    trimmer.run("../out/mc_tree.pdf", title='Middle Chinese Tones')
