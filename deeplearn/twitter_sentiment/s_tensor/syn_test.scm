
(define hidden-states 128)
(define seq-length 256)

(define (make_rnn_unit hidden-state)
    (multi-rnn-cell
      (gru-cell hidden-states)
      (gru-cell hidden-states)
      (gru-cell hidden-states))

(define-model Tweet2vec-BidirectionalGRU
  (input shape: (None seq-length))
  (one-hot _ depth: embedding-dimension axis: -1)
  (let ((forward-units (make-rnn-unit hidden-states))
        (backward-units (make-rnn-unit hidden-states)))
    (let ((rnn-unwound (dynamic-rnn forward-units backward-units)))
      (let ((output (index rnn-unwound 0))
            (states (index rnn-unwound 1)))
        (let ((fw-output (reshape (slice (index output 0) [None None] [-1 None])
                                  [-1 hidden-states]))
              (bw-output (reshape (slice (index output 1) [None None] [-1 None])
                                        [-1 hidden-states])))
            (let ((fw (variable hidden-states embedding-dimension))
                  (bw (variable hidden-states embedding-dimension)))
              (fully-connected (+ (* bw-output bw) (* fw-output fw)) num-classes)))))))
