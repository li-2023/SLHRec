import re
from predicate import Predicate, PRED_DICT
from constants import TYPE_SET, const_dict, Fact
from formula import Atom, Formula
from os.path import join as joinpath
from os.path import isfile
from utils import iterline
from cmd_args import cmd_args


def preprocess_large(dataset):

    dataroot = "data/" + dataset

    fact_path_ls = [joinpath(dataroot, 'Fact_data/facts.txt'),
                    joinpath(dataroot, 'Fact_data/train.txt')]
    query_path = joinpath(dataroot, 'Fact_data/test.txt')
    pred_path = joinpath(dataroot, 'Fact_data/relations.txt')
    const_path = joinpath(dataroot, 'Fact_data/entities.txt')
    valid_path = joinpath(dataroot, 'Fact_data/valid.txt')

    rule_path = joinpath(dataroot, 'Fact_data/rules.txt')



    TYPE_SET.update(['type'])


    for line in iterline(const_path):
        const_dict.add_const('type', line)


    for line in iterline(pred_path):
        PRED_DICT[line] = Predicate(line, ['type', 'type'])


    fact_ls = []
    for fact_path in fact_path_ls:
        for line in iterline(fact_path):
            parts = line.split('\t')

            assert len(parts) == 3, print(parts)

            e1, pred_name, e2 = parts

            assert const_dict.has_const('type', e1) and const_dict.has_const('type', e2)
            assert pred_name in PRED_DICT

            fact_ls.append(Fact(pred_name, [e1, e2], 1))

    

    valid_ls = []
    for line in iterline(valid_path):
        parts = line.split('\t')

        assert len(parts) == 3, print(parts)

        e1, pred_name, e2 = parts

        assert pred_name in PRED_DICT

        valid_ls.append(Fact(pred_name, [e1, e2], 1))


    query_ls = []
    for line in iterline(query_path):
        parts = line.split('\t')

        assert len(parts) == 3, print(parts)

        e1, pred_name, e2 = parts


        assert pred_name in PRED_DICT

        query_ls.append(Fact(pred_name, [e1, e2], 1))


    rule_ls = []
    strip_items = lambda ls: list(map(lambda x: x.strip(), ls))
    first_atom_reg = re.compile(r'([\d.]+) (!?)([^(]+)\((.*)\)')
    atom_reg = re.compile(r'(!?)([^(]+)\((.*)\)')
    for line in iterline(rule_path):

        atom_str_ls = strip_items(line.split(' v '))
        assert len(atom_str_ls) > 1, 'rule length must be greater than 1, but get %s' % line

        atom_ls = []
        rule_weight = 0.0
        for i, atom_str in enumerate(atom_str_ls):
            if i == 0:
                m = first_atom_reg.match(atom_str)
                assert m is not None, 'matching atom failed for %s' % atom_str
                rule_weight = float(m.group(1))
                neg = m.group(2) == '!'
                pred_name = m.group(3).strip()
                var_name_ls = strip_items(m.group(4).split(','))
            else:
                m = atom_reg.match(atom_str)
                assert m is not None, 'matching atom failed for %s' % atom_str
                neg = m.group(1) == '!'
                pred_name = m.group(2).strip()
                var_name_ls = strip_items(m.group(3).split(','))

            atom = Atom(neg, pred_name, var_name_ls, PRED_DICT[pred_name].var_types)
            atom_ls.append(atom)

        rule = Formula(atom_ls, rule_weight)
        rule_ls.append(rule)

    return fact_ls, rule_ls, valid_ls, query_ls



