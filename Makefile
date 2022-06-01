SHELL := /bin/bash
PARETO_DIR := pareto-probing
UD_DIR := pareto-probing/data/ud/ud-treebanks-v2.5
PARETO_PROCESSED_DIR := $(PARETO_DIR)/data/processed
PARETO_CONVERTED_DIR := $(PARETO_DIR)/data/probekit
UD_EXTRACT_LANGUAGE := ^.*\/([a-z]+)(_[a-zA-Z]+)?$$
FASTTEXT_DIR := $(PARETO_DIR)/data/fasttext
MULTILINGUAL_BERT_EMBEDDING := bert-base-multilingual-cased
MULTINLI_DIR := data/multinli_1.0

# Detect Linux vs OSX and presence of appropraite binaries
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
SPLIT = split
SED = sed
endif

ifeq ($(UNAME_S),Darwin)
ifeq (, $(shell which gsplit))
$(error "GSPLIT MISSING: You don't have gsplit installed. You need to install it, e.g., 'brew install coreutils'.")
endif
ifeq (, $(shell which gsed))
$(error "GSED MISSING: You don't have gsed installed. You need to install it, e.g., 'brew install gnu-sed'.")
endif
SPLIT = gsplit
SED = gsed
endif

# General rules
install: data/tags.yaml
	git submodule init
	git submodule update
	cp config.default.yml config.yml

data/tags.yaml:
	mkdir -p data
	cd data && wget https://raw.githubusercontent.com/unimorph/um-canonicalize/master/um_canonicalize/tags.yaml

data: process_pareto_data process_nli_data process_morpho_data process_superglue_data

data_token: process_morpho_data
data_arc: process_pareto_data
data_sentence: process_nli_data process_superglue_data

clean_data:
	rm -rf data/
	rm -rf $(PARETO_DIR)/data

# Pre-run: Download & extract fastText
fasttext_file_bin = $(FASTTEXT_DIR)/cc.$(1).300.bin
fasttext_file_bin_gz = $(FASTTEXT_DIR)/cc.$(1).300.bin.gz
fasttext_bins = $(call fasttext_file_bin,en) $(call fasttext_file_bin,tr) $(call fasttext_file_bin,ar) $(call fasttext_file_bin,mr) \
	$(call fasttext_file_bin,de) $(call fasttext_file_bin,zh)
fasttext_gzs = $(call fasttext_file_bin_gz,en) $(call fasttext_file_bin_gz,tr) $(call fasttext_file_bin_gz,ar) $(call fasttext_file_bin_gz,mr) \
	$(call fasttext_file_bin_gz,de) $(call fasttext_file_bin_gz,zh)
fasttext_url = https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$(1).300.bin.gz

$(fasttext_bins): %.bin:
	mkdir -p $(FASTTEXT_DIR)
	LANGCODE="$(subst .300,,$(subst $(FASTTEXT_DIR)/cc.,,$*))" && cd $(FASTTEXT_DIR) && curl -O $(call fasttext_url,$$LANGCODE)
	gunzip -k < $*.bin.gz > $*.bin

.PHONY: download_fasttext
download_fasttext_english: $(call fasttext_file_bin,en)
download_fasttext_turkish: $(call fasttext_file_bin,tr)
download_fasttext_arabic: $(call fasttext_file_bin,ar)
download_fasttext_marathi: $(call fasttext_file_bin,mr)
download_fasttext_german: $(call fasttext_file_bin,de)
download_fasttext_chinese: $(call fasttext_file_bin,zh)
download_fasttext: $(fasttext_bins)

### ARC DATA
# Download UD v2.5
$(UD_DIR):
	cd $(PARETO_DIR) && make get_ud

.PHONY: get_pareto_ud
get_pareto_ud: $(UD_DIR)

# Convert UD into pareto format
$(PARETO_PROCESSED_DIR)/%/train--ud.pickle.bz2: $(UD_DIR)
	cd $(PARETO_DIR) && make process LANGUAGE=$* REPRESENTATION=ud

$(PARETO_PROCESSED_DIR)/%/train--bert.pickle.bz2: $(UD_DIR)
	cd $(PARETO_DIR) && make process LANGUAGE=$* REPRESENTATION=bert

$(PARETO_PROCESSED_DIR)/%/train--fast.pickle.bz2: $(UD_DIR) download_fasttext_%
	cd $(PARETO_DIR) && make process LANGUAGE=$* REPRESENTATION=fast

pareto_processed_files = $(PARETO_PROCESSED_DIR)/$(1)/train--ud.pickle.bz2 $(PARETO_PROCESSED_DIR)/$(1)/train--fast.pickle.bz2 \
	$(PARETO_PROCESSED_DIR)/$(1)/train--bert.pickle.bz2
process_pareto_english: $(call pareto_processed_files,english)
process_pareto_turkish: $(call pareto_processed_files,turkish)
process_pareto_arabic: $(call pareto_processed_files,arabic)
process_pareto_marathi: $(call pareto_processed_files,marathi)
process_pareto_german: $(call pareto_processed_files,german)
process_pareto_chinese: $(call pareto_processed_files,chinese)
process_pareto: process_pareto_english process_pareto_turkish process_pareto_arabic process_pareto_marathi process_pareto_german \
	process_pareto_chinese

# Convert pareto format into probekit format
pareto_converted_files = $(PARETO_CONVERTED_DIR)/dep_label-$(1)-fast-train.pkl $(PARETO_CONVERTED_DIR)/dep_label-$(1)-bert-train.pkl
$(PARETO_CONVERTED_DIR)/dep_label-%-fast-train.pkl: $(PARETO_PROCESSED_DIR)/%/train--fast.pickle.bz2 $(PARETO_PROCESSED_DIR)/%/train--ud.pickle.bz2
	cd $(PARETO_DIR) && python -u src/h02_learn/convert_data.py --language $* --representation fast
$(PARETO_CONVERTED_DIR)/dep_label-%-bert-train.pkl: $(PARETO_PROCESSED_DIR)/%/train--bert.pickle.bz2 $(PARETO_PROCESSED_DIR)/%/train--ud.pickle.bz2
	cd $(PARETO_DIR) && python -u src/h02_learn/convert_data.py --language $* --representation bert
convert_pareto_english: $(call pareto_converted_files,english)
convert_pareto_turkish: $(call pareto_converted_files,turkish)
convert_pareto_arabic: $(call pareto_converted_files,arabic)
convert_pareto_marathi: $(call pareto_converted_files,marathi)
convert_pareto_german: $(call pareto_converted_files,german)
convert_pareto_chinese: $(call pareto_converted_files,chinese)
convert_pareto: convert_pareto_english convert_pareto_turkish convert_pareto_arabic convert_pareto_marathi convert_pareto_german \
	convert_pareto_chinese
process_pareto_data: convert_pareto


### MORPHOSYNTAX DATA
# Convert UD to UM
ud_paths_pattern = $(UD_DIR)/UD_$(1)/$(2)-$(3)-train.conllu $(UD_DIR)/UD_$(1)/$(2)-$(3)-dev.conllu $(UD_DIR)/UD_$(1)/$(2)-$(3)-test.conllu
ud_paths_pattern_um = $(call ud_paths_pattern,$(1),$(2),um)
ud_paths_pattern_ud = $(call ud_paths_pattern,$(1),$(2),ud)

# FROM PARETO REPOSITORY:
# 'english': 'UD_English-EWT/en_ewt-ud-%s.conllu',
# 'turkish': 'UD_Turkish-IMST/tr_imst-ud-%s.conllu'
# 'marathi': 'UD_Marathi-UFAL/mr_ufal-ud-%s.conllu'
# 'arabic': 'UD_Arabic-PADT/ar_padt-ud-%s.conllu'
# 'german': 'UD_German-GSD/de_gsd-ud-%s.conllu'
# 'chinese': 'UD_Chinese-GSDSimp/zh_gsdsimp-ud-%s.conllu',
um_paths_selected_languages = $(call ud_paths_pattern_um,English-EWT,en_ewt) \
	$(call ud_paths_pattern_um,Turkish-IMST,tr_imst) \
	$(call ud_paths_pattern_um,Marathi-UFAL,mr_ufal) \
	$(call ud_paths_pattern_um,Arabic-PADT,ar_padt) \
	$(call ud_paths_pattern_um,German-GSD,de_gsd) \
	$(call ud_paths_pattern_um,Chinese-GSDSimp,zh_gsdsimp)

# ud_all_candidates = $(wildcard $(UD_DIR)/*/*-ud-*.conllu)
# ud_all_targets = $(subst -ud-,-um-,$(ud_conversion_candidates))

$(subst -um-,-ud-,$(um_paths_selected_languages)): %: get_pareto_ud
$(filter %-um-train.conllu,$(um_paths_selected_languages)): %-um-train.conllu: %-ud-train.conllu
	MATCHED_LANGUAGE="$(shell echo '$*' | $(SED) -r -e 's/$(UD_EXTRACT_LANGUAGE)/\1/')" && \
					 cd ud-compatibility/UD_UM && pwd && python marry.py convert --ud ../../$< --lang $$MATCHED_LANGUAGE
$(filter %-um-dev.conllu,$(um_paths_selected_languages)): %-um-dev.conllu: %-ud-dev.conllu
	MATCHED_LANGUAGE="$(shell echo '$*' | $(SED) -r -e 's/$(UD_EXTRACT_LANGUAGE)/\1/')" && \
					 cd ud-compatibility/UD_UM && pwd && python marry.py convert --ud ../../$< --lang $$MATCHED_LANGUAGE
$(filter %-um-test.conllu,$(um_paths_selected_languages)): %-um-test.conllu: %-ud-test.conllu
	MATCHED_LANGUAGE="$(shell echo '$*' | $(SED) -r -e 's/$(UD_EXTRACT_LANGUAGE)/\1/')" && \
					 cd ud-compatibility/UD_UM && pwd && python marry.py convert --ud ../../$< --lang $$MATCHED_LANGUAGE

.PHONY: clean_um convert_ud_to_um
clean_um:
	rm $(UD_DIR)/**/*-um-*
convert_ud_to_um: $(um_paths_selected_languages)

# Generate files
# 'english': 'UD_English-EWT/en_ewt-ud-%s.conllu',
# 'turkish': 'UD_Turkish-IMST/tr_imst-ud-%s.conllu'
# 'marathi': 'UD_Marathi-UFAL/mr_ufal-ud-%s.conllu'
# 'arabic': 'UD_Arabic-PADT/ar_padt-ud-%s.conllu'
# 'german': 'UD_German-GSD/de_gsd-ud-%s.conllu'
# 'chinese': 'UD_Chinese-GSDSimp/zh_gsdsimp-ud-%s.conllu',
processed_morpho_filename = $(UD_DIR)/UD_$(1)/$(2)-um-$(4)-$(3).pkl
processed_morpho_files_bert = $(call processed_morpho_filename,$(1),$(2),$(MULTILINGUAL_BERT_EMBEDDING),train) \
							  $(call processed_morpho_filename,$(1),$(2),$(MULTILINGUAL_BERT_EMBEDDING),dev) \
							  $(call processed_morpho_filename,$(1),$(2),$(MULTILINGUAL_BERT_EMBEDDING),test)
processed_morpho_files_fast = $(call processed_morpho_filename,$(1),$(2),cc.$(3).300.bin,train) \
							  $(call processed_morpho_filename,$(1),$(2),cc.$(3).300.bin,dev) \
							  $(call processed_morpho_filename,$(1),$(2),cc.$(3).300.bin,test)

process_morpho_data: morpho_english morpho_turkish morpho_marathi morpho_arabic morpho_german morpho_chinese

morpho_english: morpho_english_bert morpho_english_fasttext
morpho_english_bert: $(call processed_morpho_files_bert,English-EWT,en_ewt)
morpho_english_fasttext: $(call processed_morpho_files_fast,English-EWT,en_ewt,en)

# TODO: Having $(um_paths_selected_languages) in the prerequisites means that if we add a new language, it wants to reprocess everything again
#		This should be language-specific
$(call processed_morpho_files_bert,English-EWT,en_ewt): $(um_paths_selected_languages) data/tags.yaml
	python preprocess_treebank.py UD_English-EWT --bert $(MULTILINGUAL_BERT_EMBEDDING)
$(call processed_morpho_files_fast,English-EWT,en_ewt,en): $(um_paths_selected_languages) data/tags.yaml $(call fasttext_file_bin,en)
	python preprocess_treebank.py UD_English-EWT --fast $(call fasttext_file_bin,en)

morpho_turkish: morpho_turkish_bert morpho_turkish_fasttext
morpho_turkish_bert: $(call processed_morpho_files_bert,Turkish-IMST,tr_imst)
morpho_turkish_fasttext: $(call processed_morpho_files_fast,Turkish-IMST,tr_imst,tr)
$(call processed_morpho_files_bert,Turkish-IMST,tr_imst): $(um_paths_selected_languages) data/tags.yaml
	python preprocess_treebank.py UD_Turkish-IMST --bert $(MULTILINGUAL_BERT_EMBEDDING)
$(call processed_morpho_files_fast,Turkish-IMST,tr_imst,tr): $(um_paths_selected_languages) data/tags.yaml $(call fasttext_file_bin,tr)
	python preprocess_treebank.py UD_Turkish-IMST --fast $(call fasttext_file_bin,tr)

morpho_marathi: morpho_marathi_bert morpho_marathi_fasttext
morpho_marathi_bert: $(call processed_morpho_files_bert,Marathi-UFAL,mr_ufal)
morpho_marathi_fasttext: $(call processed_morpho_files_fast,Marathi-UFAL,mr_ufal,mr)
$(call processed_morpho_files_bert,Marathi-UFAL,mr_ufal): $(um_paths_selected_languages) data/tags.yaml
	python preprocess_treebank.py UD_Marathi-UFAL --bert $(MULTILINGUAL_BERT_EMBEDDING)
$(call processed_morpho_files_fast,Marathi-UFAL,mr_ufal,mr): $(um_paths_selected_languages) data/tags.yaml $(call fasttext_file_bin,mr)
	python preprocess_treebank.py UD_Marathi-UFAL --fast $(call fasttext_file_bin,mr)

morpho_arabic: morpho_arabic_bert morpho_arabic_fasttext
morpho_arabic_bert: $(call processed_morpho_files_bert,Arabic-PADT,ar_padt)
morpho_arabic_fasttext: $(call processed_morpho_files_fast,Arabic-PADT,ar_padt,ar)
$(call processed_morpho_files_bert,Arabic-PADT,ar_padt): $(um_paths_selected_languages) data/tags.yaml
	python preprocess_treebank.py UD_Arabic-PADT --bert $(MULTILINGUAL_BERT_EMBEDDING)
$(call processed_morpho_files_fast,Arabic-PADT,ar_padt,ar): $(um_paths_selected_languages) data/tags.yaml $(call fasttext_file_bin,ar)
	python preprocess_treebank.py UD_Arabic-PADT --fast $(call fasttext_file_bin,ar)

morpho_german: morpho_german_bert morpho_german_fasttext
morpho_german_bert: $(call processed_morpho_files_bert,German-GSD,de_gsd)
morpho_german_fasttext: $(call processed_morpho_files_fast,German-GSD,de_gsd,de)
$(call processed_morpho_files_bert,German-GSD,de_gsd): $(um_paths_selected_languages) data/tags.yaml
	python preprocess_treebank.py UD_German-GSD --bert $(MULTILINGUAL_BERT_EMBEDDING)
$(call processed_morpho_files_fast,German-GSD,de_gsd,de): $(um_paths_selected_languages) data/tags.yaml $(call fasttext_file_bin,de)
	python preprocess_treebank.py UD_German-GSD --fast $(call fasttext_file_bin,de)

morpho_chinese: morpho_chinese_bert morpho_chinese_fasttext
morpho_chinese_bert: $(call processed_morpho_files_bert,Chinese-GSDSimp,zh_gsdsimp)
morpho_chinese_fasttext: $(call processed_morpho_files_fast,Chinese-GSDSimp,zh_gsdsimp,zh)
$(call processed_morpho_files_bert,Chinese-GSDSimp,zh_gsdsimp): $(um_paths_selected_languages) data/tags.yaml
	python preprocess_treebank.py UD_Chinese-GSDSimp --bert $(MULTILINGUAL_BERT_EMBEDDING)
$(call processed_morpho_files_fast,Chinese-GSDSimp,zh_gsdsimp,zh): $(um_paths_selected_languages) data/tags.yaml $(call fasttext_file_bin,zh)
	python preprocess_treebank.py UD_Chinese-GSDSimp --fast $(call fasttext_file_bin,zh)

### NLI DATA
data/multinli_1.0.zip:
	mkdir -p data/
	cd data && wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip

$(MULTINLI_DIR): data/multinli_1.0.zip
	cd data && unzip multinli_1.0.zip

$(MULTINLI_DIR)/multinli_1.0_dev.jsonl:
	cd $(MULTINLI_DIR) && $(SPLIT) -n l/2 multinli_1.0_dev_matched.jsonl && mv xaa multinli_1.0_dev.jsonl && mv xab multinli_1.0_test.jsonl

nli_data_repr = $(MULTINLI_DIR)/multinli_1.0_train_$(1).pkl $(MULTINLI_DIR)/multinli_1.0_dev_$(1).pkl $(MULTINLI_DIR)/multinli_1.0_test_$(1).pkl
nli_targets_except_ft = $(call nli_data_repr,bert) $(call nli_data_repr,albert) $(call nli_data_repr,roberta) $(call nli_data_repr,xlnet) $(call nli_data_repr,t5)
nli_targets = $(nli_targets_except_ft) $(call nli_data_repr,fasttext)
$(nli_targets_except_ft): $(MULTINLI_DIR)/multinli_1.0_%.pkl: $(MULTINLI_DIR)/multinli_1.0_dev.jsonl
	SPLIT=$(word 1,$(subst _, ,$*)) && REPR=$(word 2,$(subst _, ,$*)) && python generate_nli_data.py $$SPLIT $(MULTINLI_DIR)/multinli_1.0_$*.pkl --batch-size 64 --model $$REPR
$(call nli_data_repr,fasttext): $(MULTINLI_DIR)/multinli_1.0_%.pkl: $(MULTINLI_DIR)/multinli_1.0_dev.jsonl $(call fasttext_file_bin,en)
	SPLIT=$(word 1,$(subst _, ,$*)) && REPR=$(word 2,$(subst _, ,$*)) && python generate_nli_data.py $$SPLIT $(MULTINLI_DIR)/multinli_1.0_$*.pkl --batch-size 64 --model $$REPR


.PHONY: get_nli_data process_nli_data clean_nli
get_nli_data: $(MULTINLI_DIR) $(MULTINLI_DIR)/multinli_1.0_dev.jsonl
process_nli_data: get_nli_data $(nli_targets)
clean_nli:
	rm -rf $(MULTINLI_DIR)
	rm -rf data/__MACOSX
clean_nli_processed:
	rm $(MULTINLI_DIR)/*.pkl


### SUPERGLUE DATA
# arg1: name_sensitive - BoolQ
# arg2: name_lower - boolq
define superglue_data_RULE
data/$(1).zip:
	mkdir -p data/
	cd data && wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/$(1).zip

data/$(1): | data/$(1).zip
	cd data && unzip $(1).zip

$(2)_data_repr = data/$(1)/train_$$(1).pkl data/$(1)/val_$$(1).pkl
$(2)_targets = $$(call $(2)_data_repr,bert) $$(call $(2)_data_repr,fasttext) $$(call $(2)_data_repr,roberta) $$(call $(2)_data_repr,xlnet) $$(call $(2)_data_repr,albert) $$(call $(2)_data_repr,t5)
$$(call $(2)_data_repr,bert) $$(call $(2)_data_repr,fasttext) $$(call $(2)_data_repr,roberta) $$(call $(2)_data_repr,xlnet) $$(call $(2)_data_repr,albert) $$(call $(2)_data_repr,t5): data/$(1)/%.pkl: | data/$(1)
	SPLIT=$$(word 1,$$(subst _, ,$$*)) && REPR=$$(word 2,$$(subst _, ,$$*)) && python generate_superglue_data.py $(2) $$$$SPLIT data/$(1)/$$*.pkl --batch-size 16 --model $$$$REPR

.PHONY: get_$(2)_data process_$(2)_data clean_$(2)
get_$(2)_data: data/$(1)
process_$(2)_data: get_$(2)_data $$($(2)_targets)
clean_$(2):
	rm -rf data/$(1)
clean_$(2)_processed:
	rm data/$(1)/*.pkl
endef
$(eval $(call superglue_data_RULE,BoolQ,boolq))
$(eval $(call superglue_data_RULE,CB,cb))
$(eval $(call superglue_data_RULE,COPA,copa))
$(eval $(call superglue_data_RULE,RTE,rte))

process_superglue_data: process_boolq_data process_cb_data process_copa_data process_rte_data
clean_superglue_data: clean_boolq clean_cb clean_copa clean_rte
