import pulp as plp
import pandas as pd


def extend(a, b):
    len_dif = len(a) - len(b)
    if len_dif > 0:
        b += MISSED_CHAR * len_dif
    if len_dif < 0:
        a += MISSED_CHAR * (-len_dif)
    return a, b


def get_extention(reference_sentence, predicted_sentence):
    reference_sentence, predicted_sentence = extend(reference_sentence, predicted_sentence)
    I = range(len(reference_sentence))

    X = {(i, j): plp.LpVariable(f'X_{i}_{j}', cat="Binary") for i in I for j in I}

    error_model = plp.LpProblem(name="MILP_Model")

    # connection can exist if two char are the same
    [error_model.addConstraint(plp.LpConstraint(
        e=X[(i, j)],
        sense=plp.LpConstraintLE,
        rhs=1 if reference_sentence[i] == predicted_sentence[j] else 0
    ))
        for i in I for j in I
    ];

    # each char in a can assing to only one char in b or nothing
    [error_model.addConstraint(plp.LpConstraint(
        e=plp.lpSum([X[(i, j)] for i in I]),
        sense=plp.LpConstraintLE,
        rhs=1
    ))
        for j in I
    ];

    # each char in b can assing to only one char in a or nothing
    [error_model.addConstraint(plp.LpConstraint(
        e=plp.lpSum([X[(i, j)] for j in I]),
        sense=plp.LpConstraintLE,
        rhs=1
    ))
        for i in I
    ];

    # avoid cross connections clockwise
    [error_model.addConstraint(plp.LpConstraint(
        e=plp.lpSum([X[(i, j)], X[ip, jm]]),
        sense=plp.LpConstraintLE,
        rhs=1
    ))
        for i in I for j in I for ip in I if ip > i for jm in I if jm < j
    ];

    # avoid cross connections counter-clockwise
    [error_model.addConstraint(plp.LpConstraint(
        e=plp.lpSum([X[(i, j)], X[im, jp]]),
        sense=plp.LpConstraintLE,
        rhs=1
    ))
        for i in I for j in I for im in I if im < i for jp in I if jp > j
    ]

    objective = plp.lpSum([X[(i, j)] for i in I for j in I])
    error_model.sense = plp.LpMaximize
    error_model.setObjective(objective)
    error_model.solve()

    x_df = pd.DataFrame.from_dict(X, orient="index",
                                  columns=["variable_object"])
    x_df["solution_value"] = x_df["variable_object"].apply(lambda item: item.varValue)

    sol_df = x_df[x_df['solution_value'] > 0]
    a_dic = {i: c for i, c in enumerate(reference_sentence)}
    b_dic = {i: c for i, c in enumerate(predicted_sentence)}
    for (i, j), _ in sol_df.iterrows():
        if i == j:
            continue
        elif i > j:
            tmp_dic = dict()
            for p, c in a_dic.items():
                if p <= i:
                    tmp_dic[p - i + j] = c
                else:
                    tmp_dic[p] = c
            a_dic = tmp_dic
            tmp_dic = dict()
            for p, c in b_dic.items():
                if p < j:
                    tmp_dic[p - i + j] = c
                else:
                    tmp_dic[p] = c
            b_dic = tmp_dic
        elif i < j:
            tmp_dic = dict()
            for p, c in a_dic.items():
                if p < i:
                    tmp_dic[p - j + i] = c
                else:
                    tmp_dic[p] = c
            a_dic = tmp_dic
            tmp_dic = dict()
            for p, c in b_dic.items():
                if p <= j:
                    tmp_dic[p - j + i] = c
                else:
                    tmp_dic[p] = c
            b_dic = tmp_dic
    min_ind = min(list(a_dic.keys()) + list(b_dic.keys()));
    max_ind = max(list(a_dic.keys()) + list(b_dic.keys()));
    for p in range(min_ind, max_ind + 1):
        if p not in a_dic.keys():
            a_dic[p] = MISSED_CHAR
        if p not in b_dic.keys():
            b_dic[p] = MISSED_CHAR
    b_ext = "".join([b_dic[p] for p in range(min_ind, max_ind + 1) if
                     not ((b_dic[p] == MISSED_CHAR) and (a_dic[p] == MISSED_CHAR))])
    a_ext = "".join([a_dic[p] for p in range(min_ind, max_ind + 1) if
                     not ((b_dic[p] == MISSED_CHAR) and (a_dic[p] == MISSED_CHAR))])
    return a_ext, b_ext


def find_error(a, b):
    sol = dict()
    for Y, (w, s) in enumerate([[3, 2], [6, 2], [7, 3], [8, 3], [9, 5]]):
        window_size = w
        stride = s
        a_cor = a
        b_cor = b
        start = 0
        while (start + window_size) < len(a_cor) and (start + window_size) < len(b_cor):
            a_fin = a_cor[:start]
            b_fin = b_cor[:start]
            a_win = a_cor[start:(start + window_size)]
            a_res = a_cor[(start + window_size):]
            b_win = b_cor[start:(start + window_size)]
            b_res = b_cor[(start + window_size):]
            a_ext, b_ext = get_extention(a_win, b_win)
            a_cor = a_fin + a_ext + a_res
            b_cor = b_fin + b_ext + b_res
            start += stride
        a_fin = a_cor[:start]
        b_fin = b_cor[:start]
        a_win = a_cor[start:]
        b_win = b_cor[start:]
        a_ext, b_ext = get_extention(a_win, b_win)
        a_cor = a_fin + a_ext
        b_cor = b_fin + b_ext
        count = 0
        for i in range(len(a)):
            if a_cor[i] == b_cor[i]:
                count += 1
        score = count / len(a)
        sol[Y] = {'score': score, 'a_cor': a_cor, 'b_cor': b_cor}
    return sol


def get_error(a, b):
    sol = find_error(a, b)
    score = 0
    key = None
    for i, info in sol.items():
        if info['score'] > score:
            key = i
            score = info['score']
    if key is not None:
        return sol[key]


def get_map(a, b):
    a_cor, b_cor = get_pairs(a, b)
    return zip(a_cor, b_cor)


def get_maps(a_list, b_list):
    return [list(get_map(a, b)) for a, b in zip(a_list, b_list)]


def get_pairs(a, b):
    sol = get_error(a, b)
    a_cor = sol['a_cor']
    b_cor = sol['b_cor']
    return a_cor, b_cor


def get_ngram_error(a, b, n):
    a_wins = []
    b_wins = []
    for i in range(len(a) - n + 1):
        a_win = a[i:(i + n)]
        b_win = b[i:(i + n)]
        a_wins.append(a_win)
        b_wins.append(b_win)
    return a_wins, b_wins
