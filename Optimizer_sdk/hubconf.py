from CollapseGrammarOptimizer_vGH1_0 import CollapseGrammarOptimizer_vGH1

dependencies = ['torch']

def collapse_grammar_optimizer_vgh1(lr=1e-3):
    return CollapseGrammarOptimizer_vGH1(lr=lr)