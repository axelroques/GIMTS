# Grammar Induction on Multivariate Time Series (GIMTS)

Python implementation of a time series motif discovery method for multivariate time series using grammar induction (or grammatical inference). The motifs discovered have variable lengths: inter-motif subsequences have variable lengths, but the intra-motif subsequences are also not restricted to have identical length.

Inspired from _Li et. al. (2013)_ but extended to multivariate time series. The algorithm has four main steps:

- Transform each time series into a symbolic representation using the SAX algorithm (_Lin et. al., 2003_). Rather than discretizing the whole time series, subsequences of length _n_ are extracted from the time series, normalized and converted into a SAX word, with a stride of _k_. Each subsequence is thus discretized individually using SAX, and all of these SAX words are concatenated to form one single SAX phrase.
- From this SAX phrase, numerosity reduction is employed: if a word occurs multiple times consecutively, we only keep its first occurrence. _E.g._, the phrase 'ABC ABC ABC ABB ABB ACB ABB' becomes 'ABC ABB ACB ABB'. Numerosity reduction is the key that makes variable-length motif discovery possible.
- Using the 'reduced' phrase, a grammar induction algorithm is used. Contrary to _Li et. al. (2013)_, we used the _RePair_ algorithm from _Larsson & Moffat (1999)_. _RePair_ is an offline algorithm and as such could lead to better grammar rules compared to _Sequitur_ because all of the data is directly available. However, we note that if we were interested in streaming data _Sequitur_ would be a better choice. In this work, we sorted the rules in descending order of length and then in increasing order of occurrence - _i.e._ the most interesting rules are those that are the longest and that appear more frequently. Note that we are using the extension of _RePair_ to multivariate time series as proposed in (https://github.com/axelroques/M-RePair).
- Finally, in order to visualize the grammar rules - _i.e._ the motifs found - we need to reverse the rules on the reduced SAX representation of the time series back to the original, real-valued time series.

---

## Requirements

**Mandatory**:

- pandas
- numpy
- M-RePair (https://github.com/axelroques/M-RePair)
- SAX (https://github.com/axelroques/SAX)

**Optional**:

- matplotlib

---

## Examples

**Processing the data**:

```python
df = pd.read_csv('your_data.csv')
gimts = GIMTS(df, w=3, a=4, n=100, k=10)
gimts.process()
```

The `gimts.sax` object contains results from the different steps of the SAX algorithm and the various SAX parameters:

- `gimts.sax.df_INT` returns the dataframe after the normalization step.
- `gimts.sax.df_PAA` returns the dataframe after the PAA step.
- `gimts.sax.df_SAX` returns the dataframe after the discretization step.
- `gimts.sax.w` returns the number of segments in the PAA - and SAX - representation (after the dimensionality reduction).
- `gimts.sax.a` returns the number of symbols in the alphabet.
- `gimts.sax.alphabet` returns the different symbols in the alphabet (determined by parameter _a_).
- `gimts.sax.breakpoints` returns the values of the different breakpoints computed to discretize the time series.

The `gimts.gi` object contains results from the _M-Repair_ algorithm. For easier access, the rule and the 'RePair-ed' phrases can be found through `gimts.rules`, `gimts.phrases` and `gimts.ranked_rules`.

**Visualizing the results**:

- 'RePair-ed' Phrases

  ```python
  gimts.show_phrases()
  ```

- Grammar rules

  ```python
  gimts.show_rules(ordered=False)
  ```

- Sorted grammar rules

  ```python
  gimts.show_rules(ordered=True)
  ```

- Motifs

  ```python
  gimts.show_motifs(i_rule=14)
  ```

Where _i_rule_ is the index of the rule in the corresponding `gi.ranked_rules` DataFrame.
