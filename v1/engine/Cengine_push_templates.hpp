public
    :  // ---- Push templates
       // -----------------------------------------------------------------------------
template <typename OP>
Chandle* push(Chandle* h0) {
  return new_handle(enqueue_for_handle(new OP(nodeof(h0))));
}

template <typename OP>
Chandle* push(Chandle* h0, Chandle* h1) {
  return new_handle(enqueue_for_handle(new OP(nodeof(h0), nodeof(h1))));
}

template <typename OP>
Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2) {
  return new_handle(
      enqueue_for_handle(new OP(nodeof(h0), nodeof(h1), nodeof(h2))));
}

template <typename OP>
Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, Chandle* h3) {
  return new_handle(enqueue_for_handle(
      new OP(nodeof(h0), nodeof(h1), nodeof(h2), nodeof(h3))));
}

template <typename OP>
Chandle* push(vector<Chandle*> v) {
  vector<Cnode*> n(v.size());
  for (int i = 0; i < v.size(); i++) {
    n[i] = v[i]->node;
  }
  return enqueue_for_handle(new OP(n));
}

template <typename OP>
Chandle* push(Chandle* h0, vector<Chandle*> v) {
  vector<Cnode*> n(v.size());
  for (int i = 0; i < v.size(); i++) {
    n[i] = v[i]->node;
  }
  return enqueue_for_handle(new OP(nodeof(h0), n));
}

// ---- 1 arg

template <typename OP, typename ARG0>
Chandle* push(const ARG0& arg0) {  // changed!!
  return new_handle(enqueue_for_handle(new OP(arg0)));
}

template <typename OP, typename ARG0>
Chandle* push(Chandle* h0, const ARG0 arg0) {
  return new_handle(enqueue_for_handle(new OP(nodeof(h0), arg0)));
}

template <typename OP, typename ARG0>
Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0) {
  return new_handle(enqueue_for_handle(new OP(nodeof(h0), nodeof(h1), arg0)));
}

template <typename OP, typename ARG0>
Chandle* push(Chandle* h0, Chandle* h1, Chandle* h2, const ARG0 arg0) {
  return new_handle(
      enqueue_for_handle(new OP(nodeof(h0), nodeof(h1), nodeof(h2), arg0)));
}

template <typename OP, typename ARG0>
Chandle* push(Chandle* h0,
              Chandle* h1,
              Chandle* h2,
              Chandle* h3,
              const ARG0 arg0) {
  return new_handle(enqueue_for_handle(
      new OP(nodeof(h0), nodeof(h1), nodeof(h2), nodeof(h3), arg0)));
}

template <typename OP, typename ARG0>
Chandle* push(Chandle* h0, vector<const Chandle*> _v1, const ARG0 arg0) {
  vector<Cnode*> v1(_v1.size());
  for (int i = 0; i < _v1.size(); i++) {
    v1[i] = nodeof(_v1[i]);
  }
  return new_handle(enqueue_for_handle(new OP(nodeof(h0), v1, arg0)));
}

// ----  2 args

template <typename OP, typename ARG0, typename ARG1>
Chandle* push(const ARG0 arg0, const ARG1 arg1) {
  return new_handle(enqueue_for_handle(new OP(arg0, arg1)));
}

template <typename OP, typename ARG0, typename ARG1>
Chandle* push(Chandle* h0, const ARG0 arg0, const ARG1 arg1) {
  return new_handle(enqueue_for_handle(new OP(nodeof(h0), arg0, arg1)));
}

template <typename OP, typename ARG0, typename ARG1>
Chandle* push(Chandle* h0, Chandle* h1, const ARG0 arg0, const ARG1 arg1) {
  return new_handle(
      enqueue_for_handle(new OP(nodeof(h0), nodeof(h1), arg0, arg1)));
}

template <typename OP, typename ARG0, typename ARG1>
Chandle* push(Chandle* h0,
              Chandle* h1,
              Chandle* h2,
              const ARG0 arg0,
              const ARG1 arg1) {
  return new_handle(enqueue_for_handle(
      new OP(nodeof(h0), nodeof(h1), nodeof(h2), arg0, arg1)));
}

// ---- 3 args

template <typename OP, typename ARG0, typename ARG1, typename ARG2>
Chandle* push(const ARG0 arg0, const ARG1 arg1, const ARG2 arg2) {
  return new_handle(enqueue_for_handle(new OP(arg0, arg1, arg2)));
}

template <typename OP, typename ARG0, typename ARG1, typename ARG2>
Chandle* push(Chandle* h0, const ARG0 arg0, const ARG1 arg1, const ARG2 arg2) {
  return new_handle(enqueue_for_handle(new OP(nodeof(h0), arg0, arg1, arg2)));
}

template <typename OP, typename ARG0, typename ARG1, typename ARG2>
Chandle* push(Chandle* h0,
              Chandle* h1,
              const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2) {
  return new_handle(
      enqueue_for_handle(new OP(nodeof(h0), nodeof(h1), arg0, arg1, arg2)));
}

template <typename OP, typename ARG0, typename ARG1, typename ARG2>
Chandle* push(Chandle* h0,
              Chandle* h1,
              Chandle* h2,
              const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2) {
  return new_handle(enqueue_for_handle(
      new OP(nodeof(h0), nodeof(h1), nodeof(h2), arg0, arg1, arg2)));
}

// ---- 4 args

template <typename OP,
          typename ARG0,
          typename ARG1,
          typename ARG2,
          typename ARG3>
Chandle* push(const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2,
              const ARG3 arg3) {
  return new_handle(enqueue_for_handle(new OP(arg0, arg1, arg2, arg3)));
}

template <typename OP,
          typename ARG0,
          typename ARG1,
          typename ARG2,
          typename ARG3>
Chandle* push(Chandle* h0,
              const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2,
              const ARG3 arg3) {
  return new_handle(
      enqueue_for_handle(new OP(nodeof(h0), arg0, arg1, arg2, arg3)));
}

template <typename OP,
          typename ARG0,
          typename ARG1,
          typename ARG2,
          typename ARG3>
Chandle* push(Chandle* h0,
              Chandle* h1,
              const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2,
              const ARG3 arg3) {
  return new_handle(enqueue_for_handle(
      new OP(nodeof(h0), nodeof(h1), arg0, arg1, arg2, arg3)));
}

template <typename OP,
          typename ARG0,
          typename ARG1,
          typename ARG2,
          typename ARG3>
Chandle* push(Chandle* h0,
              Chandle* h1,
              Chandle* h2,
              const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2,
              const ARG3 arg3) {
  return new_handle(enqueue_for_handle(
      new OP(nodeof(h0), nodeof(h1), nodeof(h2), arg0, arg1, arg2, arg3)));
}

// ---- 5 args

template <typename OP,
          typename ARG0,
          typename ARG1,
          typename ARG2,
          typename ARG3,
          typename ARG4>
Chandle* push(const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2,
              const ARG3 arg3,
              const ARG4 arg4) {
  return new_handle(enqueue_for_handle(new OP(arg0, arg1, arg2, arg3, arg4)));
}

// ---- 6 args

template <typename OP,
          typename ARG0,
          typename ARG1,
          typename ARG2,
          typename ARG3,
          typename ARG4,
          typename ARG5>
Chandle* push(const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2,
              const ARG3 arg3,
              const ARG4 arg4,
              const ARG5 arg5) {
  return new_handle(
      enqueue_for_handle(new OP(arg0, arg1, arg2, arg3, arg4, arg5)));
}

// ---- 7 args

template <typename OP,
          typename ARG0,
          typename ARG1,
          typename ARG2,
          typename ARG3,
          typename ARG4,
          typename ARG5,
          typename ARG6>
Chandle* push(const ARG0 arg0,
              const ARG1 arg1,
              const ARG2 arg2,
              const ARG3 arg3,
              const ARG4 arg4,
              const ARG5 arg5,
              const ARG6 arg6) {
  return new_handle(
      enqueue_for_handle(new OP(arg0, arg1, arg2, arg3, arg4, arg5, arg6)));
}
