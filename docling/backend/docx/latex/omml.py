"""
Office Math Markup Language (OMML)

Adapted from https://github.com/xiilei/dwml/blob/master/dwml/omml.py
On 23/01/2025
"""

import logging

import lxml.etree as ET
from pylatexenc.latexencode import UnicodeToLatexEncoder

from docling.backend.docx.latex.latex_dict import (
    ALN,
    ARR,
    BACKSLASH,
    BLANK,
    BRK,
    CHARS,
    CHR,
    CHR_BO,
    CHR_DEFAULT,
    D_DEFAULT,
    F_DEFAULT,
    FUNC,
    FUNC_PLACE,
    LIM_FUNC,
    LIM_TO,
    LIM_UPP,
    POS,
    POS_DEFAULT,
    RAD,
    RAD_DEFAULT,
    SUB,
    SUP,
    D,
    F,
    M,
    T,
)

OMML_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

_log = logging.getLogger(__name__)


def load(stream):
    """Parses an OMML stream and yields LaTeX representations.

    This function takes a file-like object containing OMML XML, parses it,
    and yields the LaTeX representation for each `<oMath>` element found.

    Args:
        stream: A file-like object containing the OMML XML.

    Yields:
        A string containing the LaTeX representation of an `<oMath>` element.
    """
    tree = ET.parse(stream)
    for omath in tree.findall(OMML_NS + "oMath"):
        yield oMath2Latex(omath)


def load_string(string):
    """Parses an OMML string and yields LaTeX representations.

    This function takes a string containing OMML XML, parses it, and yields
    the LaTeX representation for each `<oMath>` element found.

    Args:
        string: A string containing the OMML XML.

    Yields:
        A string containing the LaTeX representation of an `<oMath>` element.
    """
    root = ET.fromstring(string)
    for omath in root.findall(OMML_NS + "oMath"):
        yield oMath2Latex(omath)


def escape_latex(strs):
    """Escapes special LaTeX characters in a string.

    This function iterates through a string and prepends a backslash to any
    character that has a special meaning in LaTeX (e.g., '{', '}', '_').

    Args:
        strs: The input string to be escaped.

    Returns:
        The escaped string.
    """
    last = None
    new_chr = []
    strs = strs.replace(r"\\", "\\")
    for c in strs:
        if (c in CHARS) and (last != BACKSLASH):
            new_chr.append(BACKSLASH + c)
        else:
            new_chr.append(c)
        last = c
    return BLANK.join(new_chr)


def get_val(key, default=None, store=CHR):
    """Retrieves a value from a dictionary, with a default fallback.

    This is a helper function for looking up LaTeX representations in the
    dictionaries defined in this module.

    Args:
        key: The key to look up.
        default: The default value to return if the key is not found.
        store: The dictionary to search in.

    Returns:
        The value from the dictionary, or the default value.
    """
    if key is not None:
        return key if not store else store.get(key, key)
    else:
        return default


class Tag2Method:
    """A mixin class for converting XML tags to method calls.

    This class provides a mechanism for dispatching the processing of XML elements
    to specific methods based on their tag names. It is used as a base class
    for `oMath2Latex` and `Pr`.
    """

    def call_method(self, elm, stag=None):
        """Calls the appropriate method for a given XML element.

        This method looks up the method corresponding to the element's tag name
        in the `tag2meth` dictionary and calls it.

        Args:
            elm: The XML element to process.
            stag: The short tag name (without the namespace).

        Returns:
            The result of the called method, or `None` if no method is found.
        """
        getmethod = self.tag2meth.get
        if stag is None:
            stag = elm.tag.replace(OMML_NS, "")
        method = getmethod(stag)
        if method:
            return method(self, elm)
        else:
            return None

    def process_children_list(self, elm, include=None):
        """Processes the children of an XML element and yields the results as a list.

        Args:
            elm: The parent XML element.
            include: An optional list of tag names to include in the processing.

        Yields:
            A tuple `(tag_name, processed_result, child_element)` for each
            processed child.
        """
        for _e in list(elm):
            if OMML_NS not in _e.tag:
                continue
            stag = _e.tag.replace(OMML_NS, "")
            if include and (stag not in include):
                continue
            t = self.call_method(_e, stag=stag)
            if t is None:
                t = self.process_unknow(_e, stag)
                if t is None:
                    continue
            yield (stag, t, _e)

    def process_children_dict(self, elm, include=None):
        """Processes the children of an XML element and returns a dictionary.

        Args:
            elm: The parent XML element.
            include: An optional list of tag names to include.

        Returns:
            A dictionary mapping child tag names to their processed results.
        """
        latex_chars = dict()
        for stag, t, e in self.process_children_list(elm, include):
            latex_chars[stag] = t
        return latex_chars

    def process_children(self, elm, include=None):
        """Processes the children of an XML element and returns a concatenated string.

        Args:
            elm: The parent XML element.
            include: An optional list of tag names to include.

        Returns:
            A string containing the concatenated results of processing the children.
        """
        return BLANK.join(
            (
                t if not isinstance(t, Tag2Method) else str(t)
                for stag, t, e in self.process_children_list(elm, include)
            )
        )

    def process_unknow(self, elm, stag):
        """A placeholder for handling unknown XML tags.

        This method can be overridden by subclasses to provide custom handling
        for unrecognized elements.

        Args:
            elm: The unknown XML element.
            stag: The short tag name of the element.

        Returns:
            `None` by default.
        """
        return None


class Pr(Tag2Method):
    """A class for processing common property elements in OMML.

    This class handles the parsing of property elements (those with a "Pr" suffix),
    extracting their values and making them available as attributes.

    Attributes:
        text: The processed text content of the property element.
    """

    text = ""

    __val_tags = ("chr", "pos", "begChr", "endChr", "type")

    __innerdict = None  # can't use the __dict__

    """ common properties of element"""

    def __init__(self, elm):
        self.__innerdict = {}
        self.text = self.process_children(elm)

    def __str__(self):
        return self.text

    def __unicode__(self):
        return self.__str__(self)

    def __getattr__(self, name):
        return self.__innerdict.get(name, None)

    def do_brk(self, elm):
        self.__innerdict["brk"] = BRK
        return BRK

    def do_common(self, elm):
        stag = elm.tag.replace(OMML_NS, "")
        if stag in self.__val_tags:
            t = elm.get(f"{OMML_NS}val")
            self.__innerdict[stag] = t
        return None

    tag2meth = {
        "brk": do_brk,
        "chr": do_common,
        "pos": do_common,
        "begChr": do_common,
        "endChr": do_common,
        "type": do_common,
    }


class oMath2Latex(Tag2Method):
    """Converts an `<oMath>` element from OMML to a LaTeX string.

    This is the main class for the OMML to LaTeX conversion. It traverses the
    XML tree of an `<oMath>` element and recursively converts each sub-element
    into its LaTeX equivalent.

    Attributes:
        latex: A property that returns the final, generated LaTeX string.
    """

    _t_dict = T

    __direct_tags = ("box", "sSub", "sSup", "sSubSup", "num", "den", "deg", "e")
    u = UnicodeToLatexEncoder(
        replacement_latex_protection="braces-all",
        unknown_char_policy="keep",
        unknown_char_warning=False,
    )

    def __init__(self, element):
        """Initializes the oMath2Latex converter.

        Args:
            element: The `<oMath>` XML element to be converted.
        """
        self._latex = self.process_children(element)

    def __str__(self):
        """Returns the generated LaTeX string."""
        return self.latex.replace("  ", " ")

    def __unicode__(self):
        """Returns the generated LaTeX string."""
        return self.__str__(self)

    def process_unknow(self, elm, stag):
        """Handles unknown or directly processed tags."""
        if stag in self.__direct_tags:
            return self.process_children(elm)
        elif stag[-2:] == "Pr":
            return Pr(elm)
        else:
            return None

    @property
    def latex(self):
        """The final generated LaTeX string."""
        return self._latex

    def do_acc(self, elm):
        """Converts an accent element (`<acc>`)."""
        c_dict = self.process_children_dict(elm)
        latex_s = get_val(
            c_dict["accPr"].chr, default=CHR_DEFAULT.get("ACC_VAL"), store=CHR
        )
        return latex_s.format(c_dict["e"])

    def do_bar(self, elm):
        """Converts a bar element (`<bar>`)."""
        c_dict = self.process_children_dict(elm)
        pr = c_dict["barPr"]
        latex_s = get_val(pr.pos, default=POS_DEFAULT.get("BAR_VAL"), store=POS)
        return pr.text + latex_s.format(c_dict["e"])

    def do_d(self, elm):
        """Converts a delimiter element (`<d>`)."""
        c_dict = self.process_children_dict(elm)
        pr = c_dict["dPr"]
        null = D_DEFAULT.get("null")

        s_val = get_val(pr.begChr, default=D_DEFAULT.get("left"), store=T)
        e_val = get_val(pr.endChr, default=D_DEFAULT.get("right"), store=T)
        delim = pr.text + D.format(
            left=null if not s_val else escape_latex(s_val),
            text=c_dict["e"],
            right=null if not e_val else escape_latex(e_val),
        )
        return delim

    def do_spre(self, elm):
        """Handles the Pre-Sub-Superscript element (not currently supported)."""

    def do_sub(self, elm):
        """Converts a subscript element (`<sub>`)."""
        text = self.process_children(elm)
        return SUB.format(text)

    def do_sup(self, elm):
        """Converts a superscript element (`<sup>`)."""
        text = self.process_children(elm)
        return SUP.format(text)

    def do_f(self, elm):
        """Converts a fraction element (`<f>`)."""
        c_dict = self.process_children_dict(elm)
        pr = c_dict.get("fPr")
        if pr is None:
            # Handle missing fPr element gracefully
            _log.debug("Missing fPr element in fraction, using default formatting")
            latex_s = F_DEFAULT
            return latex_s.format(
                num=c_dict.get("num"),
                den=c_dict.get("den"),
            )
        latex_s = get_val(pr.type, default=F_DEFAULT, store=F)
        return pr.text + latex_s.format(num=c_dict.get("num"), den=c_dict.get("den"))

    def do_func(self, elm):
        """Converts a function-apply element (`<func>`), e.g., sin, cos."""
        c_dict = self.process_children_dict(elm)
        func_name = c_dict.get("fName")
        return func_name.replace(FUNC_PLACE, c_dict.get("e"))

    def do_fname(self, elm):
        """Converts a function name element (`<fName>`)."""
        latex_chars = []
        for stag, t, e in self.process_children_list(elm):
            if stag == "r":
                if FUNC.get(t):
                    latex_chars.append(FUNC[t])
                else:
                    _log.warning("Function not supported, will default to text: %s", t)
                    if isinstance(t, str):
                        latex_chars.append(t)
            elif isinstance(t, str):
                latex_chars.append(t)
        t = BLANK.join(latex_chars)
        return t if FUNC_PLACE in t else t + FUNC_PLACE  # do_func will replace this

    def do_groupchr(self, elm):
        """Converts a group-character element (`<groupChr>`)."""
        c_dict = self.process_children_dict(elm)
        pr = c_dict["groupChrPr"]
        latex_s = get_val(pr.chr)
        return pr.text + latex_s.format(c_dict["e"])

    def do_rad(self, elm):
        """Converts a radical element (`<rad>`), e.g., square root."""
        c_dict = self.process_children_dict(elm)
        text = c_dict.get("e")
        deg_text = c_dict.get("deg")
        if deg_text:
            return RAD.format(deg=deg_text, text=text)
        else:
            return RAD_DEFAULT.format(text=text)

    def do_eqarr(self, elm):
        """Converts an equation array element (`<eqArr>`)."""
        return ARR.format(
            text=BRK.join(
                [t for stag, t, e in self.process_children_list(elm, include=("e",))]
            )
        )

    def do_limlow(self, elm):
        """Converts a lower-limit element (`<limLow>`)."""
        t_dict = self.process_children_dict(elm, include=("e", "lim"))
        latex_s = LIM_FUNC.get(t_dict["e"])
        if not latex_s:
            raise RuntimeError("Not support lim {}".format(t_dict["e"]))
        else:
            return latex_s.format(lim=t_dict.get("lim"))

    def do_limupp(self, elm):
        """Converts an upper-limit element (`<limUpp>`)."""
        t_dict = self.process_children_dict(elm, include=("e", "lim"))
        return LIM_UPP.format(lim=t_dict.get("lim"), text=t_dict.get("e"))

    def do_lim(self, elm):
        """Converts a limit element (`<lim>`)."""
        return self.process_children(elm).replace(LIM_TO[0], LIM_TO[1])

    def do_m(self, elm):
        """Converts a matrix element (`<m>`)."""
        rows = []
        for stag, t, e in self.process_children_list(elm):
            if stag == "mPr":
                pass
            elif stag == "mr":
                rows.append(t)
        return M.format(text=BRK.join(rows))

    def do_mr(self, elm):
        """Converts a matrix row element (`<mr>`)."""
        return ALN.join(
            [t for stag, t, e in self.process_children_list(elm, include=("e",))]
        )

    def do_nary(self, elm):
        """Converts an n-ary operator element (`<nary>`)."""
        res = []
        bo = ""
        for stag, t, e in self.process_children_list(elm):
            if stag == "naryPr":
                bo = get_val(t.chr, store=CHR_BO)
            else:
                res.append(t)
        return bo + BLANK.join(res)

    def process_unicode(self, s):
        """Converts a Unicode string to its LaTeX representation."""
        # s = s if isinstance(s,unicode) else unicode(s,'utf-8')
        # print(s, self._t_dict.get(s, s), unicode_to_latex(s))
        # _str.append( self._t_dict.get(s, s) )

        out_latex_str = self.u.unicode_to_latex(s)

        if (
            s.startswith("{") is False
            and out_latex_str.startswith("{")
            and s.endswith("}") is False
            and out_latex_str.endswith("}")
        ):
            out_latex_str = f" {out_latex_str[1:-1]} "

        if "ensuremath" in out_latex_str:
            out_latex_str = out_latex_str.replace("\\ensuremath{", " ")
            out_latex_str = out_latex_str.replace("}", " ")

        if out_latex_str.strip().startswith("\\text"):
            out_latex_str = f" \\text{{{out_latex_str}}} "

        return out_latex_str

    def do_r(self, elm):
        """Converts a run element (`<r>`), which contains text."""
        _str = []
        _base_str = []
        found_text = elm.findtext(f"./{OMML_NS}t")
        if found_text:
            for s in found_text:
                out_latex_str = self.process_unicode(s)
                _str.append(out_latex_str)
                _base_str.append(s)

        proc_str = escape_latex(BLANK.join(_str))
        base_proc_str = BLANK.join(_base_str)

        if "{" not in base_proc_str and "\\{" in proc_str:
            proc_str = proc_str.replace("\\{", "{")

        if "}" not in base_proc_str and "\\}" in proc_str:
            proc_str = proc_str.replace("\\}", "}")

        return proc_str

    tag2meth = {
        "acc": do_acc,
        "r": do_r,
        "bar": do_bar,
        "sub": do_sub,
        "sup": do_sup,
        "f": do_f,
        "func": do_func,
        "fName": do_fname,
        "groupChr": do_groupchr,
        "d": do_d,
        "rad": do_rad,
        "eqArr": do_eqarr,
        "limLow": do_limlow,
        "limUpp": do_limupp,
        "lim": do_lim,
        "m": do_m,
        "mr": do_mr,
        "nary": do_nary,
    }
