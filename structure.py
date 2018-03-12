import ntpath
import os
import pickle
from nltk import word_tokenize, RegexpTokenizer
import nltk.data
import re
from stop_words import get_stop_words
import pymorphy2
from tqdm import tqdm
import random








class Document:
    def __init__(self, doc_path=None):
        if doc_path:
            self.file_name = ntpath.basename(doc_path)
            doc_text = _get_text_from_file(doc_path)
        else:
            doc_text = ''
        self.title, self.preamble, numbered_text, self.attachment = _get_parts_of_text(doc_text)
        self.main_text = _get_main_text(numbered_text)

    def __repr__(self):
        s = '{0}\n{1}\n{2}\n{3}'.format(self.title, self.preamble, self.get_text_recursively(), self.attachment)
        return s

    def get_titles_of_chapters(self):
        return list(
            map(lambda x: ' '.join(str(x).lower().split()), filter(lambda x: len(str(x)) <= 100, self.main_text)))

    def get_text_recursively(self):
        return '\n'.join([chapter.get_text_recursively() for chapter in self.main_text])

    def __getitem__(self, name: str):
        """
        get chapter as Paragraph object by name
        :param name: title of chapter
        :return: chapter as Paragraph if it exist or None
        """

        titles_of_chapters = list(map(lambda x: x.lower(), self.get_titles_of_chapters()))
        index_chapter = titles_of_chapters.index(name.lower()) if name.lower() in titles_of_chapters else None
        if index_chapter is not None:
            return self.main_text[index_chapter]
        else:
            return None

    def save(self, path, file_name):
        """
        Serialisation
        :param path: directory for saving
        :param file_name: name of file
        """
        with open(os.path.join(path, file_name), 'wb') as f:
            pickle.dump(self, f, 2)

    def load(self, path):
        with open(path, 'rb') as f:
            try:
                doc = pickle.load(f)
                self.title = doc.title
                self.preamble = doc.preamble
                self.main_text = doc.main_text
                self.attachment = doc.attachment
            except Exception as e:
                print(path)
                raise e



class Paragraph:
    """
    Stores sentences of Paragraph in `sentences` object, that is a list.
    Each element of list is instance of `Sentence`
    """

    def __init__(self, text: str = ''):
        self.sentences = _row_text_to_sentences(text)
        self.children = []
        self.queue = []

    def get_sentences(self, normalize=False, tokenize=False, remove_stop_words=False):
        sentences = []
        if not self.queue:
            _depth_first_traversal(self, self.queue)
        for paragraph in self.queue:
            for sentence in paragraph.sentences:
                tmp_sentence = sentence
                if remove_stop_words:
                    tmp_sentence = sentence.remove_stop_words()
                if normalize:
                    sentences.append(tmp_sentence.normalize())
                else:
                    sentences.append(tmp_sentence)
        if tokenize:
            sentences = list(map(lambda x: x.tokenize(), sentences))
        return sentences

    def get_text_recursively(self):
        chapter_sentences = []
        _print_chapter(self, chapter_sentences)
        return '\n'.join(chapter_sentences)

    def add_sentences(self, row_text: str):
        self.sentences += _row_text_to_sentences(row_text)

    def add_children(self, paragraph):
        self.children.append(paragraph)

    def __repr__(self):
        return ' '.join([str(s) for s in self.sentences])


class Sentence:
    """
    Stores information about sentence.
    :sentence: str
    :language: default russian. required for `tokenize` method.
    """

    def __init__(self, sentence: str, language='russian'):
        self.sentence = sentence
        self.language = language
        self.weight = None
        self.named_entities = None

    def tokenize(self):
        """
        Get tokens of sentence. Uses `nltk.word_tokenize`.
        """
        return word_tokenize(self.sentence, language=self.language)

    def regexp_tokenize(self):
        tokenizer = RegexpTokenizer(r'[а-яА-ЯA-Za-z]+')
        return tokenizer.tokenize(self.sentence)

    def remove_stop_words(self):
        pattern = r'[^\w-]'
        stop_words = get_stop_words(self.language)
        tokens = self.tokenize()
        good_tokens = []
        for token in tokens:
            if not re.search(pattern, token) and token.lower() not in stop_words:
                good_tokens.append(token)
        return Sentence(' '.join(good_tokens), language=self.language)

    def set_named_entities(self, entities):
        self.named_entities = entities

    def normalize(self):
        morph = pymorphy2.MorphAnalyzer()
        tokens = self.tokenize()
        norm_tokens = []
        for token in tokens:
            norm_tokens.append(morph.parse(token)[0].normal_form)
        return Sentence(' '.join(norm_tokens), language=self.language)

    def __repr__(self):
        return self.sentence


def _get_text_from_file(doc_path):
    filename, file_extension = os.path.splitext(doc_path)
    if file_extension == '.txt':
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
    else:
        raise NotImplemented('This type of file is not support. Convert it to .txt.')
    return doc_text


def _row_text_to_sentences(text):
    text = text.replace('\t', ' ')
    text = [line.strip() for line in text.split('\n')]
    text = ' '.join(text)
    tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    special_abbr = ['млн', 'коп', 'руб']
    for abbr in special_abbr:
        tokenizer._params.abbrev_types.add(abbr)
    sentences = list(map(lambda x: Sentence((x.strip())), tokenizer.tokenize(text)))
    return sentences


def _depth_first_traversal(tree, queue, level=0):
    queue.append(tree)
    if tree.children:
        for i in range(len(tree.children)):
            _depth_first_traversal(tree.children[i], queue, level=level + 1)


def _print_chapter(tree, q, level=0):
    q.append('')
    for i in range(level):
        q[-1] += '\t'
    q[-1] += str(tree)
    if tree.children:
        for i in range(len(tree.children)):
            _print_chapter(tree.children[i], q, level + 1)


def _get_parts_of_text(doc_text):
    got_title = False
    got_preamble = False
    got_attachment = False
    found_right_attachment = False
    title = Paragraph()
    preamble = Paragraph()
    numbered_text = []
    attachment = Paragraph()
    attachment_sentences = []
    lines = list(map(lambda x: x.strip(), doc_text.split('\n')))
    for line in lines:
        if line == '':
            continue
        elif not got_title:
            title.add_sentences(line)
            got_title = True
        elif not got_preamble:
            if line.strip()[0:2] == "1." or (line.strip()[0] == "1" and 'предмет' in line.lower().strip()):
                got_preamble = True
                numbered_text.append(line)
            else:
                preamble.add_sentences(line)
        elif got_preamble and not got_attachment:
            if line.strip().lower().startswith('приложение') or (
                    'приложение' in line.lower() and len('приложение') / len(line.strip()) > 0.5):
                got_attachment = True
                attachment_sentences.append(line)
            else:
                numbered_text.append(line)
        elif got_attachment and not found_right_attachment:
            if line.strip().lower().startswith('приложение №1'):
                numbered_text += attachment_sentences
                attachment_sentences = [line]
                found_right_attachment = True
            else:
                attachment_sentences.append(line)
        elif found_right_attachment:
            attachment_sentences.append(line)
        else:
            numbered_text.append(line)
    for sentence in attachment_sentences:
        attachment.add_sentences(sentence)
    return title, preamble, numbered_text, attachment


def _get_main_text(numbered_text):
    pred_num = [0, 0, 0]
    main_text = []
    tmp_num = []
    for line in numbered_text:
        num = _get_num_from_str(line)
        if not num and tmp_num:
            pred_num, main_text = _add_paragraph(tmp_num, pred_num, line, main_text)
            tmp_num = []
            continue
        while (line and (line[0].isdigit() or line[0] == '.')) and len(line) > 0:
            line = line[1:]
        if line:
            pred_num, main_text = _add_paragraph(num, pred_num, line, main_text)
        else:
            tmp_num = num

    return main_text


def _get_num_from_str(s):
    num = ''
    for elem in s:
        if elem == '.' or elem.isdigit():
            num += elem
        else:
            break
    num = list(map(lambda x: int(x), filter(lambda x: x != '', num.split('.'))))
    return num


def _add_paragraph(num, pred_num, text, main_text):
    if not num and pred_num[1] == 0 and pred_num[2] == 0 and main_text:
        main_text[-1].add_children(Paragraph(text))
        pred_num[1] = 1
    elif len(num) == 1 and pred_num[0] == num[0] - 1:
        main_text.append(Paragraph(text))
        pred_num[0] = num[0]
        pred_num[1], pred_num[2] = 0, 0
    elif len(num) == 2 and pred_num[0] == num[0] and pred_num[1] == num[1] - 1 and main_text:
        main_text[-1].add_children(Paragraph(text))
        pred_num[1] = num[1]
        pred_num[2] = 0
    elif len(num) == 3 and pred_num[0] == num[0] and pred_num[1] == num[1] and pred_num[2] == num[2] - 1 and \
            main_text[-1].children:
        main_text[-1].children[-1].add_children(Paragraph(text))
        pred_num[2] = num[2]
    elif pred_num[0] != 0 and pred_num[1] == 0 and pred_num[2] == 0 and main_text:
        main_text[-1].add_children(Paragraph(text))
        pred_num[1] = 1
    else:
        _add_text_to_last_paragraph(pred_num, main_text, text)
    return pred_num, main_text


def _add_text_to_last_paragraph(pred_num, main_text, text):
    level = len(list(filter(lambda x: x != 0, pred_num)))
    if level == 1:
        main_text[-1].add_sentences(text)
    if level == 2:
        main_text[-1].children[-1].add_sentences(text)
    if level == 3:
        main_text[-1].children[-1].children[-1].add_sentences(text)


if __name__ == "__main__":
    # Examples of using
    # d = Document('contract1.txt')
    # parsed_doc = Document()
    # parsed_doc.load('txts_pickled/contract5153')
    # print(parsed_doc.get_titles_of_chapters())
    # print(d['предмет договора'].get_sentences(normalize=True, tokenize=False, remove_stop_words=True))


    # переименоваем документы, чтобы они были пронумерованы
    #i = 0
    #for filename in tqdm(os.listdir('txts/')):
    #    new_name = 'contract{}.txt'.format(i)
    #    os.rename('txts/' + filename, 'txts/' + new_name)
    #    i = i + 1


    # SAVE IN PICKLE FORMAT:
    # txts - folder contained 15 000 texts
    # txts_pickled - folder contained 15 000 pickled texts
    #for j in tqdm(range(15000)):
    #    d = Document('txts/contract{}.txt'.format(j))
    #    d.save('txts_pickled', 'contract{}'.format(j))


    # удаляем те документы, в которых не обнаружились главы
    #for filename in tqdm(os.listdir('txts_pickled/')):
    #    doc = Document()
    #    doc.load('txts_pickled/' + filename)
    #    if len(doc.get_titles_of_chapters()) == 0:
    #        os.remove('txts_pickled/' + filename)
    #        os.remove('txts/' + filename + '.txt')


    # к имеющимся 15000 добавляем еще 3000 документов с подменёнными абзацами
    """i = 0
    while i < 3000:
        try:
            ind_of_doc = random.randint(0, 14999)  # выбираем индекс случайного документа, откуда будем брать пункт
            print('ind_of_doc =', ind_of_doc)
            doc = Document()
            doc.load('txts_pickled/contract{}'.format(ind_of_doc))  # id_of_doc -ый документ
            chapters = doc.get_titles_of_chapters()  # получаем названия глав
            ind_of_chapter = random.randint(0, len(chapters)-1)  # выбираем индекс случайной главы
            print('ind_of_chapter =', ind_of_chapter)
            doc[chapters[ind_of_chapter]].get_sentences()
            queue = doc[chapters[ind_of_chapter]].queue  # получаем лист из пунктов в главе
            del queue[0]  # т.к. 0-ой элемент листа - название главы
            ind_of_item = random.randint(0, len(queue)-1)  # выбираем индекс случайного пункта
            print('ind_of_item =', ind_of_item)
            item = queue[ind_of_item]  # получаем необходимый пункт

            new_ind_of_doc = random.randint(0, 14999)  # новый индекс документа, в котором будем заменять пункт
            new_doc = Document()
            new_doc.load('txts_pickled/contract{}'.format(new_ind_of_doc))
            print('new doc is', new_ind_of_doc)
            new_chapters = new_doc.get_titles_of_chapters()
            print(new_chapters)
            new_ind_of_chapter = random.randint(0, len(new_chapters)-1)
            print('new_ind_of_chapter =', new_ind_of_chapter)
            new_doc[new_chapters[new_ind_of_chapter]].get_sentences()
            new_queue = new_doc[new_chapters[new_ind_of_chapter]].queue
            del new_queue[0]
            new_ind_of_item = random.randint(0, len(new_queue)-1)
            print('new_ind_of_item =', new_ind_of_item)
            new_queue[new_ind_of_item] = item
            new_doc.save('txts_pickled', 'contract{}'.format(15000 + i))
        except Exception as e:
            i = i - 1
        print(i)
        i = i + 1"""