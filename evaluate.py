def evaluate(serialization_dir):
    # construct file name
    
    if class_wise:
        file_name = "triggers.json"
    else:
        file_name = ''
        if self.indices_of_token_to_modify:
            file_name += 'pos-'
        if self.error_max == 1:
            file_name += 'error_max'
        if epoch is None:
            file_name += "init_modification"
        else:
            assert batch_idx is not None
            file_name += f"modification_epoch{epoch}_batch{batch_idx}.json"

    
    with open( os.path.join(serialization_dir, file_name), 'w') as fp:
        json.dump(modifications, fp)

    with open( os.path.join(serialization_dir, file_name), 'a') as fp:
        fp.write(output)

def save_modifications_for_squad():
    final_output = {}
    # add modified texts into modifications
    all_instances = deepcopy(self.instances)
    for instance, modification in zip(all_instances, self.modifications):
        output_dict = {}
        # origin info: passage, question, answer
        output_dict['orig_passage'] = instance.fields['metadata'].metadata["original_passage"]
        output_dict['question'] = " ".join(instance.fields['metadata'].metadata["question_tokens"])
        ## answer
        answer_start = instance.fields['span_start'].sequence_index
        answer_end = instance.fields['span_end'].sequence_index
        output_dict['answer_text'] = instance.fields['metadata'].metadata["answer_texts"]
        # answer_tokens =  instance.fields['passage'].tokens[answer_start:answer_end+1]
        # output_dict['answer_text'] = " ".join([str(token) for token in answer_tokens])
        output_dict['answer_start_position'] = answer_start
        output_dict['answer_end_position'] = answer_end   

        # info after modification
        position_to_modify, substitution = list(modification.items())[0]
        output_dict['modified_position'] = int(position_to_modify)
        output_dict['modified_word'] = str(instance.fields['passage'].tokens[int(position_to_modify)])
        output_dict['substitution_word']= substitution
        self.modify_one_example(instance, int(position_to_modify), substitution, self.input_field_name)
        output_dict['modified_passage'] = instance.fields['metadata'].metadata["original_passage"]
        output_dict['distance_to_answer'] = min(abs(answer_start-int(position_to_modify)),abs(answer_end -int(position_to_modify)))
    
        id = instance.fields['metadata'].metadata["id"]
        final_output[id] = output_dict

def generate_squad_analyzable_result( instances, modification_file_path, print_num=None):
        """
        The example of the modification format: [{"80": "the"}, {"76": "the"}, {"22": "the"}]

        # Parameters

        instances: `Iterator[Instance]`, required
        
        modification_file_path: `str`, required

        # return

        """
        with open( modification_file_path, 'r') as fp:
            modifications: List[dict] = json.load(fp)

        print_results = []
        for idx_of_dataset, modification_dict in enumerate(modifications):
            instance = instances[int(idx_of_dataset)]
            fields = instance.fields
            print_result = {'original': " ".join(str(token) for token in fields["passage"].tokens)}

            for position_to_modify, substitution in modification_dict.items():
                
                print_result['position_to_modify'] = int(position_to_modify)
                print_result['modified_word'] = str(fields['passage'].tokens[int(position_to_modify)])
                print_result['substitution_word'] = substitution
                print_result['question'] = " ".join([str(token) for token in fields['question'].tokens])

                answer_start = fields['span_start'].sequence_index
                answer_end = fields['span_end'].sequence_index
                print_result['answer_text'] = fields['metadata'].metadata["answer_texts"]
                # answer_tokens =  fields['passage'].tokens[answer_start:answer_end+1]
                # print_result['answer_text'] = " ".join([str(token) for token in answer_tokens])
                print_result['position_answer_start'] = answer_start
                print_result['position_answer_end'] = answer_end
                print_result['distance_to_answer'] = min(abs(answer_start-int(position_to_modify)),abs(answer_end -int(position_to_modify)))
                             
                break # only modify one position
            print_results.append(print_result)
            if print_num is not None and idx_of_dataset >=print_num:
                break
        return print_results