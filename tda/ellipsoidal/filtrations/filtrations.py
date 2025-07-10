from gudhi.simplex_tree import SimplexTree


def build_filtration(xs, A_list, ComplexClass, radii, max_dim=2):
    """
    For a list of radii build a global SimplexTree whose filtration value for
    every simplex is the smallest radius at which it first appears.
    """
    st = SimplexTree(None)

    # vertices appear at t = 0
    for v in range(len(xs)):
        st.insert([v], filtration=0.0)
    seen = {tuple([v]) for v in range(len(xs))}

    for eps in sorted(radii):
        comp = ComplexClass(xs, A_list, eps)
        comp.build_complex(max_dim=max_dim + 1)        # we may want tetrahedra
        simplices = comp.edges + comp.triangles + comp.tetrahedra

        # helper to convert (i,j,...) into sorted tuple
        for s in simplices:
            key = tuple(sorted(s))
            if key not in seen:
                st.insert(key, filtration=eps)
                seen.add(key)

    # st.initialize_filtration()-- we got a warning this is deprecated
    diag = st.persistence()                            # default field F₂
    return st, diag
