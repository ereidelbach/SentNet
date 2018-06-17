![Project Logo](https://raw.githubusercontent.com/ereidelbach/Images/master/SentNet.png)

----

# Project Overview

SentNet is an award winning tool crafted in response to the Office of the Director of National Intelligence (ODNI) Analytic Integrity and Standards (AIS) staff's [Xtend Challenge (Machine Evaluation of Analytic Products)](https://www.innocentive.com/ar/challenge/9934078).

Submissions to the challenge were judged against their ability to evaluate analytic products by meeting the following requirements:
* Properly describe the quality and credibility of underlying sources, data and methodologies.
* Properly express and explain uncertainties associated with major analytic judgments.
* Properly distinguish between underlying intelligence information and analysts' assumptions and judgments.
* Incorporate analysis of alternatives
* Demonstrate relavance to customers and address implications and opportunities.
* Use clear and logical argumentation.
* Explain change to or consistency of analytic judgments.
* Make sound judgments and assertions.
* Incorporate effective visual information where appopriate.

SentNet was created by [Taylor Corbett](https://github.com/data4d) and [Eric Reidelbach](https://github.com/ereidelbach).

----

# SentNet Description

SentNet is a next generation document analysis tool that utilizes advanced part-of-speech
tagging and graph analysis techniques to rapidly evaluate and numerically score
intelligence products against one (or multiple) criteria with no intervention while allowing
analysts to easily validate scores using an intuitive user interface.

----

# Resources:

TBD

# System Resources:

SetNet was developed on a system with the following resources. When attempting to run SentNet it is recommended that you have a system with similar specifications:

* Operating System: Windows 10
* Processor: Intel Core i7-7700 - 3.6 Ghz
* RAM: 16 GB
* Hard Drive: 10 GB of Hard Drive Space Available

# Python Packages

The following packages are required to run SentNet's in Python:

* argparse
* docx
* itertools
* networkx
* nltk
* nltk - wordnet (requires a separate download using the nltk downloader)
* nlkt - stopwords (requires a separate download using the nltk downloader)
* nltk - word_tokenizer (requires a separate download using the nltk downloader)
* nltk - PunktSentenceTokenizer (requires a separate download using the nltk downloader)
* numpy
* os
* pandas
* re
* sklearn
* sys
* xml.etree.ElementTree
* zipfile


