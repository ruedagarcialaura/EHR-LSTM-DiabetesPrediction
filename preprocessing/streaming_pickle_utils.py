# -*- coding: utf-8 -*-
"""
Shared helpers for saving/loading large {patient_id: DataFrame} dictionaries
without needing extra memory to serialize/deserialize "everything at once".

Why this exists: pickling a single huge dict object (tens of thousands of
patients, each with a sizeable DataFrame -- especially now that DX adds ~500
extra rows per patient) can throw a MemoryError purely from the serialization
overhead, even when the underlying data technically fits in RAM. Writing (and
reading) one (patient_id, dataframe) pair at a time avoids that overhead: each
pickle.dump()/pickle.load() call only ever has to handle one patient's worth
of data.

On-disk format: first a single int (patient count), then that many
(patient_id, dataframe) tuples, each as its own pickle object in the same
file stream.
"""

import pickle
import tqdm


def save_streamed_patient_dict(patient_data, path, desc="Saving"):
    """Save a {patient_id: DataFrame} dict incrementally, one patient at a time."""
    with open(path, "wb") as f:
        pickle.dump(len(patient_data), f)
        for pid in tqdm.tqdm(list(patient_data.keys()), desc=desc, colour='blue'):
            pickle.dump((pid, patient_data[pid]), f)


def load_streamed_patient_dict(path, desc="Loading"):
    """Load a dict saved with save_streamed_patient_dict back into a regular dict.

    Also transparently loads pickles saved the OLD way (one big dict in a
    single pickle.dump call), so this is a safe drop-in replacement for
    `pickle.load()` everywhere in the pipeline -- it auto-detects the format.
    """
    with open(path, "rb") as f:
        first_obj = pickle.load(f)
        if isinstance(first_obj, dict):
            # Old-style file: a single dict was dumped directly.
            return first_obj
        # New-style file: first_obj is the patient count, followed by
        # (patient_id, dataframe) tuples.
        n_patients = first_obj
        patient_data = {}
        for _ in tqdm.tqdm(range(n_patients), desc=desc, colour='blue'):
            pid, df = pickle.load(f)
            patient_data[pid] = df
        return patient_data
