# construct vocabulary
        ## from data
        instance_generator = (
            instance
            for key, data_loader in data_loaders.items()
            for instance in data_loader.iter_instances()
        )
        vocabulary_ = Vocabulary.from_files(directory=vocab_dir)
        ## from pre-trained transformers
        for field in next(data_loaders['train'].iter_instances()).fields.values():
            from allennlp.data.fields import TextField
            from allennlp.data.token_indexers import PretrainedTransformerIndexer
            if type(field) == TextField:
                for indexer in  field._token_indexers.values():
                    if type(indexer) == PretrainedTransformerIndexer:
                        indexer._add_encoding_to_vocabulary_if_needed(vocabulary_)