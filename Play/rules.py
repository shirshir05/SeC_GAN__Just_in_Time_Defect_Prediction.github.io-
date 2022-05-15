from abc import ABC, abstractmethod


class Move:
    def can_apply(self, x):
        return True

    def apply(self, x, key=None):
        features_change = self.affect_feature(key, x)
        for key in features_change:
            if key in x:
                x[key] += features_change[key]
        return x

    @abstractmethod
    def affect_feature(self, key=None, x=None):
        pass

    def apply_if_can(self, x, key=None):
        if self.can_apply(self, x):
            self.apply(self, x, key)
            return True
        return False


class AddWhileUnreachable(Move):
    # https://github.com/shirshir05/test_play/commit/88e4b1fdd774fe030736ae16ffa444e609f5d913
    def affect_feature(self, key=None, x=None):
        return {'delta_WhileStatement': 1, 'delta_Literal': 1, 'delta_BreakStatement': 1, 'delta_BlockStatement': 1,
                'ast_diff_loopAdd': 1, 'delta_NOS': 2, 'delta_NLE': 1, 'delta_NL': 1}


class AddSwitchStatement(Move):
    # https://github.com/shirshir05/test_play/commit/22bdb6ea1826ec11d1c5abe65da413eca223badd
    def affect_feature(self, key=None, x=None):  # todo: AddSwitchStatement
        return {'delta_SwitchStatement': 1, 'delta_Literal': 2, 'delta_SwitchStatementCase': 1,
                'ast_diff_condBranCaseAdd': 1, 'delta_NOS': 1, 'delta_NLE': 1, 'delta_NL': 1}


class ForStatementWithMember(Move):
    # TODO: Assume there is Member (No features are an indication of Member)
    # https://github.com/shirshir05/test_play/commit/f043bf544688adadf029719c5673d0ef97957d17
    def affect_feature(self, key=None, x=None):
        return {'delta_BinaryOperation': 1, 'delta_ForStatement': 1, 'delta_MemberReference': 2,
                'delta_BlockStatement': 1, 'delta_ForControl': 1, 'ast_diff_loopAdd': 1, 'delta_NOS': 1, 'delta_NLE': 1,
                'delta_NL': 1}


class AddIfMember(Move):
    # TODO: Assume there is Member (No features are an indication of Member)
    # https://github.com/shirshir05/test_play/commit/01395251c52795846996288bb6f0b1f152108bb4
    def affect_feature(self, key=None, x=None):
        return {'delta_BinaryOperation': 1, 'delta_Statement': 1, 'delta_IfStatement': 1, 'delta_MemberReference': 2,
                'ast_diff_condBranIfAdd': 1, 'delta_NOS': 2, 'delta_NLE': 1, 'delta_NL': 1}


class AddIfMember_v2(Move):
    # TODO: Assume there is Member (No features are an indication of Member)
    # https://github.com/shirshir05/test_play/commit/b841f7662726c6cc4c9ecc60fee476b67834b59d
    def affect_feature(self, key=None, x=None):
        return {"delta_BinaryOperation": 1, "delta_IfStatement": 1, "delta_MemberReference": 2,
                "ast_diff_condBranIfAdd": 1, "delta_NOS": 2, "delta_NLE": 1, "delta_NL": 1,
                'delta_Statement': 1, }


class AddWhileMember(Move):
    # TODO: Assume there is Member (No features are an indication of Member)
    # https://github.com/shirshir05/test_play/commit/f4cfdd86af4649910c3f9617abf6215b227ad305
    def affect_feature(self, key=None, x=None):
        return {"delta_MemberReference": 2, " delta_BinaryOperation": 1, 'delta_WhileStatement': 1,
                "ast_diff_loopAdd": 1, "delta_NOS": 1, "delta_NLE": 1, "delta_NL": 1,
                "delta_BlockStatement": 1,
                }


class AddWhileMemberV2(Move):
    # TODO: Assume there is Member (No features are an indication of Member)
    # https://github.com/shirshir05/test_play/commit/2f46fb445e3fe0ad5a4d11594ac89d310abc9085
    def affect_feature(self, key=None, x=None):
        return {'delta_BinaryOperation': 1, 'delta_WhileStatement': 1, 'delta_Statement': 1, 'delta_MemberReference': 2,
                'ast_diff_loopAdd': 1, 'delta_NOS': 2, 'delta_NLE': 1, 'delta_NL': 1}


class AddCommaNewLine(Move):
    # https://github.com/shirshir05/test_play/commit/eb384afe20aeee3dd8bdd5ebaf6eec4b51b812dd
    def affect_feature(self, key=None, x=None):
        return {'delta_Statement': 1, "delta_LLOC": 1, "delta_TLLOC": 1, "delta_NOS": 1,
                "used_added_lines+used_removed_lines": 1, "used_added_lines-used_removed_lines": 1}


# class AddCommaSameLine(Move):
#     # https://github.com/shirshir05/test_play/commit/2a21fcdd72ac29323a5928b0b3983d931110183c
#     def affect_feature(self, key=None):
#         return {'delta_Statement': 1, "delta_NOS": 1}


class AddEmptySwitch(Move):
    # https://github.com/shirshir05/test_play/commit/86e3fb5e0644478d00351603a0cc33cf0edf366d
    def affect_feature(self, key=None, x=None):
        return {"delta_Literal": 1, "delta_SwitchStatement": 1,
                "delta_NOS": 1, "delta_NLE": 1}


class AddComment(Move):
    # https://github.com/shirshir05/test_play/commit/3ab65bd2928324544936886c7a0ef52514a6d4c0
    def affect_feature(self, key=None, x=None):
        return {"delta_CLOC": 1, "delta_TCLOC": 1, "used_added_lines+used_removed_lines": 2}


class AddCommentNewLine(Move):
    # https://github.com/shirshir05/test_play/commit/eb94e61eddb212be470bd8243b0ff2712832585a
    def affect_feature(self, key=None, x=None):
        return {"delta_CLOC": 1, "delta_TCLOC": 1, "delta_DLOC": 1}


class CallFunctionWithoutParam(AddSwitchStatement):
    # TODO: assumption - have method that not get parameter
    # https://github.com/shirshir05/test_play/commit/6dfe1a0d410e1496728576add0f76e36068aa67c
    def affect_feature(self, key=None, x=None):
        return {**AddSwitchStatement.affect_feature(key), **{"delta_NOS": 1, "delta_MethodInvocation": 1,
                                                             "delta_StatementExpression": 1, "ast_diff_mcAdd": 1,
                                                             "ast_diff_mcParValChange": 2, }}


# class MethodCallSwitchCase(AddSwitchStatement):
#     # TODO: assumption - have method that not get parameter
#     # https://github.com/shirshir05/test_play/commit/8d781588df9f60dfc98719569991cab40bebba8d
#     def can_apply(self, x):
#         return "ast_diff_mcRem" in x and x["ast_diff_mcRem"] > 0
#
#     def affect_feature(self, key=None, x=None):
#         return {**AddSwitchStatement.affect_feature(key), **{"ast_diff_mcRem": -1,
#                                                              "delta_StatementExpression": 1,
#                                                              "delta_MethodInvocation": 1,
#                                                              "ast_diff_mcAdd": 1, "ast_diff_mcParValChange": 1,
#                                                              "delta_NOS": 1}}


class ChangeIfToFor(Move):
    # https://github.com/shirshir05/test_play/commit/aff68e735c71129a557ae0c2d9fca1eae3f8dd4a
    def can_apply(self, x):
        return ("delta_IfStatement" in x and x['delta_IfStatement'] > 0) or \
               ("parent_IfStatement" in x and x['parent_IfStatement'] > 0)

    def affect_feature(self, key=None, x=None):
        return {
            "delta_ForStatement": 1,  "delta_ForControl": 1,
            "delta_IfStatement": -1, "ast_diff_condBranIfAdd": -1,
            "delta_BreakStatement": 1,
            "ast_diff_loopAdd": 1, "delta_NOS": 1}


class ChangeIfToWhile(Move):
    # https://github.com/shirshir05/test_play/commit/9859a0ab110cd7779a0ccb3025d85b20924de98b
    def can_apply(self, x):
        return ("delta_IfStatement" in x and x['delta_IfStatement'] > 0) or \
               ("parent_IfStatement" in x and x['parent_IfStatement'] > 0)

    def affect_feature(self, key=None, x=None):
        return {"delta_WhileStatement": 1, "delta_IfStatement": -1, "delta_BreakStatement": 1,
                'ast_diff_condBranIfAdd': -1, "ast_diff_loopAdd": 1, "delta_NOS": 1}


class IfWithCallFunctionParam(Move):
    # TODO: assumption - have method that get one parameter
    # https://github.com/shirshir05/test_play/commit/951f9f0f8fa252260e3a3c8574b091d6e0c30190
    def affect_feature(self, key=None, x=None):
        return {"delta_MethodInvocation": 1, "delta_BinaryOperation": 1, "delta_StatementExpression": 1,
                "delta_IfStatement": 1, "delta_Literal": 2,
                "delta_BlockStatement": 1,
                "ast_diff_condBranIfAdd": 1, "ast_diff_mcAdd": 1,
                "ast_diff_mcParAdd": 1, "ast_diff_mcParValChange": 2, "ast_diff_objInstAdd": 1, "delta_NOS": 2,
                "delta_NLE": 1, "delta_NL": 1, }


class SwitchWithCallFunctionParam(AddSwitchStatement):
    # TODO: assumption - have method that get one parameter
    # https://github.com/shirshir05/test_play/commit/951f9f0f8fa252260e3a3c8574b091d6e0c30190
    def affect_feature(self, key=None, x=None):
        return {**AddSwitchStatement.affect_feature(key), **{"delta_MethodInvocation": 1,
                                                             "delta_StatementExpression": 1, "ast_diff_mcAdd": 1,
                                                             "ast_diff_mcParAdd": 1,
                                                             "ast_diff_mcParValChange": 2, "ast_diff_objInstAdd": 1,
                                                             'delta_NOS': 1}}


class AddEmptyLine(Move):
    # https://github.com/shirshir05/test_play/commit/981376c5389cdc47836745109e99cc72355bac6d
    def affect_feature(self, key=None, x=None):
        return {"delta_LOC": 1, "delta_TLOC": 1}


class AddMethod(Move):
    # https://github.com/shirshir05/test_play/commit/d05717ff95e599cab41ccc15698b8dfeff119e5e
    def affect_feature(self, key=None, x=None):
        return {"ast_diff_mdAdd": 1, "delta_LOC": 1, "delta_TLLOC": 1, "delta_TLOC": 1, "delta_NLM": 1,
                "delta_TNLPM": 1,
                "delta_NM": 1, "delta_NLPM": 1, "delta_TNM": 1, "delta_RFC": 1, "delta_NPM": 1, "delta_TNPM": 1,
                "delta_TNLM": 1,
                "used_added_lines+used_removed_lines": 1, "used_added_lines-used_removed_lines": 1}


class AddReturn(AddSwitchStatement):
    # https://github.com/shirshir05/test_play/commit/196d3df93e6304115e9d56ef90951d6e2f5136e1
    def affect_feature(self, key=None, x=None):
        return {**AddSwitchStatement.affect_feature(key), **{"delta_NOS": 1,
                                                             "ast_diff_retBranchAdd": 1, "delta_ReturnStatement": 1}}


class AddAssignWithSwitch(AddSwitchStatement):
    # https://github.com/shirshir05/test_play/commit/ea52d3bf7ec7267b089c27c2da8d336ff9ddb37d
    def affect_feature(self, key=None, x=None):
        return {**AddSwitchStatement.affect_feature(key), **{"delta_NOS": 1,
                                                             "delta_Literal": 1, "delta_Assignment": 1,
                                                             "delta_StatementExpression": 1, "delta_MemberReference": 1,
                                                             "ast_diff_assignAdd": 1}}


class AddVar(Move):
    # https://github.com/shirshir05/test_play/commit/bb4395f54435747756c57c4a0dc0f8b6af55ce66
    def affect_feature(self, key=None, x=None):
        return {"delta_LocalVariableDeclaration": 1, "delta_Literal": 1, "delta_VariableDeclarator": 1,
                "ast_diff_assignAdd": 1, "ast_diff_objInstMod": 1,
                "used_added_lines+used_removed_lines": 2, "ast_diff_varAdd": 1, "delta_NOS": 1}


class AddElseWithBlock(Move):
    def can_apply(self, x):
        return ("delta_IfStatement" in x and x['delta_IfStatement'] > 0) or \
               ("parent_IfStatement" in x and x['parent_IfStatement'] > 0)

    # https://github.com/shirshir05/test_play/commit/00e430b082331cbf611ab45b2bb0afc910372f90
    def affect_feature(self, key=None, x=None):
        return {'delta_Literal': 1, 'delta_IfStatement': 1, 'delta_Statement': 1, 'delta_BlockStatement': 1,
                'ast_diff_condBranIfElseAdd': 1, 'ast_diff_condBranElseAdd': 1, 'delta_NOS': 2, 'delta_NL': 1}


class AddIfFalse(Move):
    # https://github.com/shirshir05/test_play/commit/8971a661af935cb97d754ee84b7532d81fcf5d2f
    def affect_feature(self, key=None, x=None):
        return {'delta_Literal': 1, 'delta_IfStatement': 1, 'delta_BlockStatement': 1, 'ast_diff_condBranIfAdd': 1,
                'delta_NOS': 1, 'delta_NLE': 1, 'delta_NL': 1}


class AddIfFalseCommaDot(Move):
    # https://github.com/shirshir05/test_play/commit/dbf54af0b5bc0c9dff42e3aaba51e0ee542403d2
    def affect_feature(self, key=None, x=None):
        return {'delta_Literal': 1, 'delta_Statement': 1, 'delta_IfStatement': 1, 'ast_diff_condBranIfAdd': 1,
                'delta_NOS': 2, 'delta_NLE': 1, 'delta_NL': 1}


class AddIfAndElse(Move):
    # https://github.com/shirshir05/test_play/commit/d8b1fa658dd5e6fb507ff9648d38d4644b819a67
    def affect_feature(self, key=None, x=None):
        return {'delta_Literal': 1, 'delta_IfStatement': 1, 'delta_BlockStatement': 2, 'ast_diff_condBranIfElseAdd': 1,
                'ast_diff_condBranElseAdd': 1, 'delta_NOS': 1, 'delta_NLE': 1, 'delta_NL': 1}


class ThrowException(AddSwitchStatement):
    # https://github.com/shirshir05/test_play/commit/9b0ad8452ea16a41a63bef476726a3015be19cba
    def affect_feature(self, key=None, x=None):
        return {**AddSwitchStatement.affect_feature(key), **{"delta_NOS": 3, "delta_NLE": 1, "delta_NL": 1,
                                                             "ast_diff_exThrowsAdd": 1, "ast_diff_exTryCatchAdd": 1,
                                                             "ast_diff_objInstAdd": 1, "delta_ThrowStatement": 1,
                                                             "delta_Statement": 1, "delta_CatchClause": 1,
                                                             "delta_TryStatement": 1}}


class AddPublicAttribute(Move):
    # Todo: Assume have class
    # https://github.com/shirshir05/test_play/commit/4797cfce48f6c24b8b4c8f0b748485ad023c76b9
    def affect_feature(self, key=None, x=None):
        return {"ast_diff_objInstMod": 1, "delta_LOC": 1, "delta_LLOC": 1,
                "delta_TLLOC": 1, "delta_TLOC": 1,
                "delta_NPA": 1, "delta_TNLA": 1,
                "delta_NLA": 1, "delta_TNLPA": 1,
                "delta_TNA": 1, "delta_NLPA": 1, "delta_NA": 1, "delta_TNPA": 1,
                "used_added_lines+used_removed_lines": 1, "used_added_lines-used_removed_lines": 1}


class AddContinue(AddSwitchStatement):
    # https://github.com/shirshir05/test_play/commit/128493bae17787046cc52233411e1ef26dad3b4b
    def affect_feature(self, key=None, x=None):
        return {**AddSwitchStatement.affect_feature(key), **{"delta_Literal": 1, "delta_ContinueStatement": 1,
                                                             "delta_WhileStatement": 1,
                                                             "ast_diff_loopAdd": 1, "delta_NOS": 2,
                                                             "delta_NLE": 1, "delta_NL": 1}}


# https://github.com/shirshir05/SPAT_rules
# https://github.com/Santiago-Yu/SPAT

class IfDividing(Move):
    # rule 9
    def can_apply(self, x):
        return "ast_diff_condExpExpand" in x and x["ast_diff_condExpExpand"] > 2

    def affect_feature(self, key=None, x=None):
        return {'delta_BinaryOperation': -1, 'delta_BlockStatement': 1, 'delta_IfStatement': 1,
                'ast_diff_condBranIfAdd': 1, 'ast_diff_condExpExpand': -2, 'delta_LOC': 2, 'delta_LLOC': 2,
                'delta_TLLOC': 2, 'delta_NOS': 1, 'delta_TLOC': 2, 'used_added_lines+used_removed_lines': 2,
                'used_added_lines-used_removed_lines': 2}


class InfixExpressionDividing(Move):
    # rule 8
    def can_apply(self, x):
        return "ast_diff_condExpExpand" in x and x["ast_diff_condExpExpand"] > 2

    def affect_feature(self, key=None, x=None):
        return {'delta_MemberReference': 1, 'delta_LocalVariableDeclaration': 1, 'delta_VariableDeclarator': 1,
                'ast_diff_assignAdd': 1, 'ast_diff_condExpExpand': -3, 'ast_diff_objInstMod': 2, 'ast_diff_varAdd': 1,
                'delta_LOC': 1, 'delta_LLOC': 1, 'delta_TLLOC': 1, 'delta_NOS': 1, 'delta_TLOC': 1,
                'used_added_lines+used_removed_lines': 1, 'used_added_lines-used_removed_lines': 1}


class AddAssignemnt2EqualAssignment(Move):
    # TODO: Assume there is Member (No features are an indication of Member)
    def affect_feature(self, key=None, x=None):
        return {'delta_BinaryOperation': 1, 'delta_MemberReference': 1}


class PP2AddAssignment(Move):
    # rule 6
    def can_apply(self, x):
        return "delta_Literal" in x and x['delta_Literal'] > 0

    def affect_feature(self, key=None, x=None):
        return {'delta_Literal': 1, 'delta_Assignment': 1, 'ast_diff_assignAdd': 1}


class For2While(Move):
    # rule 6
    def can_apply(self, x):
        return ("delta_ForStatement" in x and x['delta_ForStatement'] > 0 and
                "delta_ForControl" in x and x['delta_ForControl'] > 0 and
                "delta_VariableDeclaration" in x and x['delta_VariableDeclaration'] > 0) or \
               ("parent_ForStatement" in x and x['parent_ForStatement'] > 0 and
                "parent_ForControl" in x and x['parent_ForControl'] > 0 and
                "parent_VariableDeclaration" in x and x['parent_VariableDeclaration'] > 0)

    def affect_feature(self, key=None, x=None):
        return {'delta_ForStatement': -1, 'delta_StatementExpression': 1, 'delta_LocalVariableDeclaration': 1,
                'delta_WhileStatement': 1, 'delta_ForControl': -1, 'delta_VariableDeclaration': -1, 'delta_LOC': 2,
                'delta_LLOC': 2, 'delta_TLLOC': 2, 'delta_TLOC': 2, 'used_added_lines+used_removed_lines': 2,
                'used_added_lines-used_removed_lines': 2}


class While2For(Move):
    # rule 6
    def can_apply(self, x):
        return ("delta_WhileStatement" in x and x['delta_WhileStatement'] > 0) or \
               ("parent_WhileStatement" in x and x['parent_WhileStatement'] > 0)

    def affect_feature(self, key=None, x=None):
        return {'delta_ForStatement': 1, 'delta_WhileStatement': -1, 'delta_ForControl': 1}


class ReverseIfElse(Move):
    # rule 6
    def can_apply(self, x):
        return "ast_diff_condBranIfAdd" in x and x['ast_diff_condBranIfAdd'] > 0

    def affect_feature(self, key=None, x=None):
        return {'delta_Statement': 1, 'ast_diff_condBranIfAdd': -1, 'ast_diff_condBranIfElseAdd': 1,
                'ast_diff_condBranElseAdd': 1, 'delta_LOC': 2, 'delta_LLOC': 2, 'delta_TLLOC': 2,
                'delta_NOS': 1, 'delta_TLOC': 2, 'used_added_lines+used_removed_lines': 2,
                'used_added_lines-used_removed_lines': 2}


class SingleIF2ConditionalExp(Move):
    # rule 4
    def can_apply(self, x):
        return "delta_BlockStatement" in x and x['delta_BlockStatement'] > 0 and \
               "delta_IfStatement" in x and x['delta_IfStatement'] > 0 and \
               "ast_diff_condBranIfAdd" in x and x['ast_diff_condBranIfAdd'] > 0 and \
               "delta_LOC" in x and x['delta_LOC'] > 0 and \
               "delta_LLOC" in x and x['delta_LOC'] > 0

    def affect_feature(self, key=None, x=None):
        return {'delta_MemberReference': 1, 'delta_TernaryExpression': 1, 'delta_BlockStatement': -1,
                'delta_IfStatement': -1, 'ast_diff_condBranIfAdd': -1, 'ast_diff_condBranIfElseAdd': 1, 'delta_LOC': -1,
                'delta_LLOC': -1, 'delta_TLLOC': -1, 'delta_NOS': -1, 'delta_TLOC': -1,
                'used_added_lines+used_removed_lines': -1, 'used_added_lines-used_removed_lines': -1}


class Case2IfElse(Move):
    # rule 17
    def can_apply(self, x):
        return "delta_SwitchStatementCase" in x and x['delta_BlockStatement'] > 3 and \
               "delta_BreakStatement" in x and x['delta_BreakStatement'] > 2 and \
               "delta_SwitchStatement" in x and x['delta_SwitchStatement'] > 0 and \
               "ast_diff_condBranCaseAdd" in x and x['ast_diff_condBranCaseAdd'] > 3 and \
               "delta_LOC" in x and x['delta_LOC'] > 3 and \
               "delta_LLOC" in x and x['delta_LLOC'] > 3 and \
               "delta_TLLOC" in x and x['delta_TLLOC'] > 3 and \
               "delta_NOS" in x and x['delta_NOS'] > 0 and \
               "delta_TLOC" in x and x['delta_TLOC'] > 3 and \
               "used_added_lines+used_removed_lines" in x and x['used_added_lines+used_removed_lines'] > 3 and \
               "used_added_lines-used_removed_lines" in x and x['used_added_lines-used_removed_lines'] > 3

    def affect_feature(self, key=None, x=None):
        return {'delta_SwitchStatementCase': -4, 'delta_IfStatement': 3, 'delta_BlockStatement': 4,
                'delta_BreakStatement': -3, 'delta_Literal': 2, 'delta_ArraySelector': 2, 'delta_BinaryOperation': 3,
                'delta_MemberReference': 2, 'delta_SwitchStatement': -1, 'ast_diff_condBranIfElseAdd': 3,
                'ast_diff_condBranElseAdd': 3, 'ast_diff_condBranCaseAdd': -4, 'delta_LOC': -4, 'delta_LLOC': -4,
                'delta_TLLOC': -4, 'delta_NOS': -1, 'delta_NL': 1, 'delta_TLOC': -4,
                'used_added_lines+used_removed_lines': -4, 'used_added_lines-used_removed_lines': -4}


# class SwitchStringEqual(Move):
#     # rule 15
#     def affect_feature(self, key=None, x=None):
#         return {'delta_Literal': 1, 'delta_BinaryOperation': 2, 'ast_diff_condExpExpand': 2}


# class VarDeclarationDividing(Move):
#     # rule 13
#     def affect_feature(self, key=None, x=None):
#         return {'delta_LocalVariableDeclaration': 1, 'delta_LOC': 1, 'delta_LLOC': 1, 'delta_TLLOC': 1,
#                 'delta_TLOC': 1, 'used_added_lines+used_removed_lines': 1, 'used_added_lines-used_removed_lines': 1}


class LoopIfContinue2Else(Move):
    def can_apply(self, x):
        return "delta_ContinueStatement" in x and x['delta_ContinueStatement'] > 0 and \
               "ast_diff_condBranIfAdd" in x and x['ast_diff_condBranIfAdd'] > 0 and \
               "delta_NOS" in x and x['delta_NOS'] > 0

    # rule 11
    def affect_feature(self, key=None, x=None):
        return {'delta_BlockStatement': 2, 'delta_ContinueStatement': -1, 'ast_diff_condBranIfAdd': -1,
                'ast_diff_condBranIfElseAdd': 1, 'ast_diff_condBranElseAdd': 1, 'delta_LOC': 2, 'delta_LLOC': 2,
                'delta_TLLOC': 2, 'delta_NOS': -1, 'delta_TLOC': 2, 'used_added_lines+used_removed_lines': 2,
                'used_added_lines-used_removed_lines': 2}
