\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{is2}[1997/01/23]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions
\LoadClass[a4paper]{report}
%\RequirePackage[dvips]{graphicx}

\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}


\ProcessOptions

\input{styles/nsize1\@ptsize.clo}

%--------------
% Neue Makros |
%--------------
% Matrikelnummer des Kandidaten
\def\matnr#1{\gdef\@matnr{#1}}
\gdef\@matnr{}
%
% Betreuer der Arbeit (wiss. Mitarbeiter)
\def\betreuer#1{\gdef\@betreuer{#1}}
\gdef\@betreuer{}
%
%
% Unterschrift 
\def\signature#1{\gdef\@signature{#1}}
\gdef\@signature{}
%
% Titel des Autors
\def\authortitel#1{\gdef\@authortitel{#1}}
\gdef\@authortitel{}
%
% Typ der Arbeit: Diplom oder Studienarbeit
\def\typ#1{\gdef\@typ{#1}}
\gdef\@typ{}
%
\def\monat#1{\gdef\@monat{#1}}
\gdef\@monat{}%










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  BEGIN OF Stuff copied from IEEEtran style to have better references        %
% (e.g. URLs are working now for MISC bibtex entries!!!!!)                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% class option conditionals
\newif\ifCLASSOPTIONonecolumn       \CLASSOPTIONonecolumnfalse
\newif\ifCLASSOPTIONtwocolumn       \CLASSOPTIONtwocolumntrue

\newif\ifCLASSOPTIONoneside         \CLASSOPTIONonesidetrue
\newif\ifCLASSOPTIONtwoside         \CLASSOPTIONtwosidefalse

\newif\ifCLASSOPTIONfinal           \CLASSOPTIONfinaltrue
\newif\ifCLASSOPTIONdraft           \CLASSOPTIONdraftfalse
\newif\ifCLASSOPTIONdraftcls        \CLASSOPTIONdraftclsfalse
\newif\ifCLASSOPTIONdraftclsnofoot  \CLASSOPTIONdraftclsnofootfalse

\newif\ifCLASSOPTIONpeerreview      \CLASSOPTIONpeerreviewfalse
\newif\ifCLASSOPTIONpeerreviewca    \CLASSOPTIONpeerreviewcafalse

\newif\ifCLASSOPTIONjournal         \CLASSOPTIONjournaltrue
\newif\ifCLASSOPTIONconference      \CLASSOPTIONconferencefalse
\newif\ifCLASSOPTIONtechnote        \CLASSOPTIONtechnotefalse

\newif\ifCLASSOPTIONnofonttune      \CLASSOPTIONnofonttunefalse

\newif\ifCLASSOPTIONcaptionsoff     \CLASSOPTIONcaptionsofffalse

\newif\ifCLASSOPTIONcompsoc         \CLASSOPTIONcompsocfalse

\newif\ifCLASSOPTIONromanappendices \CLASSOPTIONromanappendicesfalse


% class info conditionals

% indicates if pdf (via pdflatex) output
\newif\ifCLASSINFOpdf               \CLASSINFOpdffalse


% V1.6b internal flag to show if using a4paper
\newif\if@IEEEusingAfourpaper       \@IEEEusingAfourpaperfalse



% IEEEtran class scratch pad registers
% dimen
\newdimen\@IEEEtrantmpdimenA
\newdimen\@IEEEtrantmpdimenB
% count
\newcount\@IEEEtrantmpcountA
\newcount\@IEEEtrantmpcountB
% token list
\newtoks\@IEEEtrantmptoksA



% \DeclareOption{9pt}{\def\CLASSOPTIONpt{9}\def\@ptsize{0}}
% \DeclareOption{10pt}{\def\CLASSOPTIONpt{10}\def\@ptsize{0}}
% \DeclareOption{11pt}{\def\CLASSOPTIONpt{11}\def\@ptsize{1}}
% \DeclareOption{12pt}{\def\CLASSOPTIONpt{12}\def\@ptsize{2}}



% \DeclareOption{letterpaper}{\setlength{\paperheight}{11in}%
%                             \setlength{\paperwidth}{8.5in}%
%                             \@IEEEusingAfourpaperfalse
%                             \def\CLASSOPTIONpaper{letter}%
%                             \def\CLASSINFOpaperwidth{8.5in}%
%                             \def\CLASSINFOpaperheight{11in}}
% 
% 
% \DeclareOption{a4paper}{\setlength{\paperheight}{297mm}%
%                         \setlength{\paperwidth}{210mm}%
%                         \@IEEEusingAfourpapertrue
%                         \def\CLASSOPTIONpaper{a4}%
%                         \def\CLASSINFOpaperwidth{210mm}%
%                         \def\CLASSINFOpaperheight{297mm}}
% 
% \DeclareOption{oneside}{\@twosidefalse\@mparswitchfalse
%                         \CLASSOPTIONonesidetrue\CLASSOPTIONtwosidefalse}
% \DeclareOption{twoside}{\@twosidetrue\@mparswitchtrue
%                         \CLASSOPTIONtwosidetrue\CLASSOPTIONonesidefalse}
% 
% \DeclareOption{onecolumn}{\CLASSOPTIONonecolumntrue\CLASSOPTIONtwocolumnfalse}
% \DeclareOption{twocolumn}{\CLASSOPTIONtwocolumntrue\CLASSOPTIONonecolumnfalse}
% 
% % If the user selects draft, then this class AND any packages
% % will go into draft mode.
% \DeclareOption{draft}{\CLASSOPTIONdrafttrue\CLASSOPTIONdraftclstrue
%                       \CLASSOPTIONdraftclsnofootfalse}
% % draftcls is for a draft mode which will not affect any packages
% % used by the document.
% \DeclareOption{draftcls}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclstrue
%                          \CLASSOPTIONdraftclsnofootfalse}
% % draftclsnofoot is like draftcls, but without the footer.
% \DeclareOption{draftclsnofoot}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclstrue
%                                \CLASSOPTIONdraftclsnofoottrue}
% \DeclareOption{final}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclsfalse
%                       \CLASSOPTIONdraftclsnofootfalse}
% 
% \DeclareOption{journal}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
%                         \CLASSOPTIONjournaltrue\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}
% 
% \DeclareOption{conference}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
%                            \CLASSOPTIONjournalfalse\CLASSOPTIONconferencetrue\CLASSOPTIONtechnotefalse}
% 
% \DeclareOption{technote}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
%                          \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotetrue}
% 
% \DeclareOption{peerreview}{\CLASSOPTIONpeerreviewtrue\CLASSOPTIONpeerreviewcafalse
%                            \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}
% 
% \DeclareOption{peerreviewca}{\CLASSOPTIONpeerreviewtrue\CLASSOPTIONpeerreviewcatrue
%                              \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}
% 
% \DeclareOption{nofonttune}{\CLASSOPTIONnofonttunetrue}
% 
% \DeclareOption{captionsoff}{\CLASSOPTIONcaptionsofftrue}
% 
% \DeclareOption{compsoc}{\CLASSOPTIONcompsoctrue}
% 
% \DeclareOption{romanappendices}{\CLASSOPTIONromanappendicestrue}
% 
% 
% % default to US letter paper, 10pt, twocolumn, one sided, final, journal
% \ExecuteOptions{a4paper,11pt,onecolumn,twoside,final,journal}
% % overrride these defaults per user requests
% \ProcessOptions



% Computer Society conditional execution command
\long\def\@IEEEcompsoconly#1{\relax\ifCLASSOPTIONcompsoc\relax#1\relax\fi\relax}
% inverse
\long\def\@IEEEnotcompsoconly#1{\relax\ifCLASSOPTIONcompsoc\else\relax#1\relax\fi\relax}
% compsoc conference
\long\def\@IEEEcompsocconfonly#1{\relax\ifCLASSOPTIONcompsoc\ifCLASSOPTIONconference\relax#1\relax\fi\fi\relax}
% compsoc not conference
\long\def\@IEEEcompsocnotconfonly#1{\relax\ifCLASSOPTIONcompsoc\ifCLASSOPTIONconference\else\relax#1\relax\fi\fi\relax}





%% CITATION AND BIBLIOGRAPHY COMMANDS
%%
%% V1.6 no longer supports the older, nonstandard \shortcite and \citename setup stuff
%
%
% Modify Latex2e \@citex to separate citations with "], ["
\def\@citex[#1]#2{%
  \let\@citea\@empty
  \@cite{\@for\@citeb:=#2\do
    {\@citea\def\@citea{], [}%
     \edef\@citeb{\expandafter\@firstofone\@citeb\@empty}%
     \if@filesw\immediate\write\@auxout{\string\citation{\@citeb}}\fi
     \@ifundefined{b@\@citeb}{\mbox{\reset@font\bfseries ?}%
       \G@refundefinedtrue
       \@latex@warning
         {Citation `\@citeb' on page \thepage \space undefined}}%
       {\hbox{\csname b@\@citeb\endcsname}}}}{#1}}

% V1.6 we create hooks for the optional use of Donald Arseneau's
% cite.sty package. cite.sty is "smart" and will notice that the
% following format controls are already defined and will not
% redefine them. The result will be the proper sorting of the
% citation numbers and auto detection of 3 or more entry "ranges" -
% all in IEEE style:  [1], [2], [5]--[7], [12]
% This also allows for an optional note, i.e., \cite[mynote]{..}.
% If the \cite with note has more than one reference, the note will
% be applied to the last of the listed references. It is generally
% desired that if a note is given, only one reference is listed in
% that \cite.
% Thanks to Mr. Arseneau for providing the required format arguments
% to produce the IEEE style.
\def\citepunct{], [}
\def\citedash{]--[}

% V1.7 default to using same font for urls made by url.sty
\AtBeginDocument{\csname url@samestyle\endcsname}

% V1.6 class files should always provide these
\def\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty


% Provide support for the control entries of IEEEtran.bst V1.00 and later.
% V1.7 optional argument allows for a different aux file to be specified in
% order to handle multiple bibliographies. For example, with multibib.sty:
% \newcites{sec}{Secondary Literature}
% \bstctlcite[@auxoutsec]{BSTcontrolhak}
\def\bstctlcite{\@ifnextchar[{\@bstctlcite}{\@bstctlcite[@auxout]}}
\def\@bstctlcite[#1]#2{\@bsphack
  \@for\@citeb:=#2\do{%
    \edef\@citeb{\expandafter\@firstofone\@citeb}%
    \if@filesw\immediate\write\csname #1\endcsname{\string\citation{\@citeb}}\fi}%
  \@esphack}

% V1.6 provide a way for a user to execute a command just before
% a given reference number - used to insert a \newpage to balance
% the columns on the last page
\edef\@IEEEtriggerrefnum{0}   % the default of zero means that
                              % the command is not executed
\def\@IEEEtriggercmd{\newpage}

% allow the user to alter the triggered command
\long\def\IEEEtriggercmd#1{\long\def\@IEEEtriggercmd{#1}}

% allow user a way to specify the reference number just before the
% command is executed
\def\IEEEtriggeratref#1{\@IEEEtrantmpcountA=#1%
\edef\@IEEEtriggerrefnum{\the\@IEEEtrantmpcountA}}%

% trigger command at the given reference
\def\@IEEEbibitemprefix{\@IEEEtrantmpcountA=\@IEEEtriggerrefnum\relax%
\advance\@IEEEtrantmpcountA by -1\relax%
\ifnum\c@enumiv=\@IEEEtrantmpcountA\relax\@IEEEtriggercmd\relax\fi}


\def\@biblabel#1{[#1]}

% compsoc journals left align the reference numbers
\@IEEEcompsocnotconfonly{\def\@biblabel#1{[#1]\hfill}}

% controls bib item spacing
\def\IEEEbibitemsep{0pt plus .5pt}

\@IEEEcompsocconfonly{\def\IEEEbibitemsep{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}}


\def\thebibliography#1{%
    \chapter*{\refname
      \@mkboth{%
          \MakeUppercase\refname}{\MakeUppercase\refname}}%
    \addcontentsline{toc}{chapter}{\refname}%
    % V1.6 add some rubber space here and provide a command trigger
    %\footnotesize\@IEEEcompsocconfonly{\small}\vskip 0.3\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip%
    \normalsize\@IEEEcompsocconfonly{\small}\vskip 0.3\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip%
    \list{\@biblabel{\@arabic\c@enumiv}}%
    {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep\relax
    \itemsep \IEEEbibitemsep\relax
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    \let\@IEEElatexbibitem\bibitem%
    \def\bibitem{\@IEEEbibitemprefix\@IEEElatexbibitem}%
\def\newblock{\hskip .11em plus .33em minus .07em}%
% originally:
%   \sloppy\clubpenalty4000\widowpenalty4000%
% by adding the \interlinepenalty here, we make it more
% difficult, but not impossible, for LaTeX to break within a reference.
% IEEE almost never breaks a reference (but they do it more often with
% technotes). You may get an underfull vbox warning around the bibliography,
% but the final result will be much more like what IEEE will publish.
% MDS 11/2000
\ifCLASSOPTIONtechnote\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty100%
\else\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty500\fi%
    \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  END OF Stuff copied from IEEEtran style to have better references          %
% (e.g. URLs are working now for MISC bibtex entries!!!!!)                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
















%
%-----------------------------------------------------------------
% Seitenaufbau                                                   |
%-----------------------------------------------------------------
%
\def\maketitleis2{
%
%------------------
% Titelseite      |
%------------------
%
\begin{titlepage}
\let\footnotesize\normalsize \let\footnoterule\relax \setcounter{page}{0}
\vskip -3.5cm
\evensidemargin=0.75cm
\oddsidemargin=0.25cm
\begin{center}
Institute for Integrated Signal Processing Systems (ISS)\par
RWTH Aachen University \par
Prof. Dr.-Ing. Gerd Ascheid \par
\vskip 5.0cm
{\Large \@typ\par}
\vskip 1.5em
\evensidemargin=6.0cm
\oddsidemargin=6.0cm
\begin{minipage}[t]{9.8cm}
    \begin{center}
        {\LARGE \@title \par}
    \end{center}
\end{minipage}
\vskip 3em
\evensidemargin=0.75cm
\oddsidemargin=0.25cm
{by\par}
\vskip 1.5em
%{\Large \@authortitel \\ \@author \par}
{\Large \@author \par}
\vskip 1.5em
{Matr.-No. \@matnr \par}
\vskip 1.5em
{\@monat\par}
\end{center} \par
\vskip 3.5cm
\noindent Supervisors:\par
\noindent Prof. Dr.-Ing. Gerd Ascheid\par
\noindent \@betreuer\par
\vfill
\noindent This document is for internal use only. All copyrights are controlled by the supervising chair.\\ 
Publications of any kind are only authorized with permission of the chair.\par
\end{titlepage}
    %--------------------------
    % insert empty page --- IMPORTANT: never use \begin{titlepage} ! destroys page numbering
    % (backside of title page)
    %--------------------------
    \clearpage
    \thispagestyle{empty}
    \hbox{}
    \newpage
    %--------------------------
    % insert empty sheet
    % (to prevent seeing the signature page through the not-100%-opaque title page)
    %--------------------------
    \clearpage
    \thispagestyle{empty}
    \hbox{}
    \newpage
    \thispagestyle{empty}
    \hbox{}
    \newpage
%
%--------------------------
% Seite mit Erklaerung    |
%--------------------------
%
\begin{titlepage}
\let\footnotesize\normalsize \let\footnoterule\relax \setcounter{page}{0}
\null
\vskip 9cm
\noindent
I assure that this project was accomplished by me, without any foreign
assistance except the official support of the chair. The used literature is
fully indicated in the bibliography.\par
\vskip 3em
\makebox[\textwidth][l]{Aachen, \@date}
\vskip 6em
\makebox[\textwidth][l]{(\@signature)}
\vfil
\end{titlepage}
    %--------------------------
    % insert empty page --- IMPORTANT: never use \begin{titlepage} ! destroys page numbering
    %--------------------------
    \clearpage
    \thispagestyle{empty}
    \hbox{}
    \newpage

    %---------------------------
    % insert placeholder sheet for the official thesis title document signed by professor and student 
    %---------------------------
    \clearpage
    \thispagestyle{empty}
    \hbox{}
    \vskip 4cm
    \noindent
    {\huge
        \centering FIXME:\\
        \vskip 3cm
        \centering Placeholder\\
        \vskip 5cm
        \centering Please replace this sheet by the official thesis title/task document signed by the professor and you.\\
    }
    \newpage
    \thispagestyle{empty}
    \hbox{}
    \newpage
%
%------------------------------------
% Ruecksetzen der verwendeten Makros |
%-------------------------------------
%
\setcounter{footnote}{0}
\gdef\@typ{} \gdef\@betreuer{} \gdef\@matnr{} \gdef\@authortitel{} \gdef\@author{} \gdef\@title{} \gdef\@signature{} 
\gdef\@monat{} \def\maketitleis2{}
}

\setlength{\parindent}{5mm}

\renewcommand\tableofcontents{%
    \setcounter{page}{1}
    \pagestyle{headings}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \pagenumbering{roman}
    \@starttoc{toc}
    \if@restonecol\twocolumn\fi


    %--------------------------
    % insert empty page --- IMPORTANT: never use \begin{titlepage} ! destroys page numbering
    %--------------------------
    \clearpage
    \thispagestyle{empty}
    \hbox{}
    \newpage

    %--------------------------

    \if@openright\cleardoublepage\else\clearpage\fi

    \setcounter{page}{1}
    \pagenumbering{arabic}
    }

\renewcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}}%
    \@starttoc{lof}%
    \addcontentsline{toc}{chapter}{\listfigurename}
    \if@restonecol\twocolumn\fi
    }

\newcommand{\listlistingname}{List of Listings}

\newcommand\listoflistings{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listlistingname
      \@mkboth{\MakeUppercase\listlistingname}%
              {\MakeUppercase\listlistingname}}%
    \@starttoc{lol}%
    \addcontentsline{toc}{chapter}{\listlistingname}
    \if@restonecol\twocolumn\fi
    }

\renewcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{%
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}%
    \@starttoc{lot}%
    \addcontentsline{toc}{chapter}{\listtablename}
    \if@restonecol\twocolumn\fi
    }

% \renewcommand\bibliography[1]{%
%   \if@filesw
%     \immediate\write\@auxout{\string\bibdata{#1}}%
%   \fi
%   \@input@{\jobname.bbl}
%   \addcontentsline{toc}{chapter}{\bibname}}
% 
% \newenvironment{issbibliography}[1]
%      {\chapter*{\bibname
%         \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}}%
%       \list{\@biblabel{\@arabic\c@enumiv}}%
%            {\settowidth\labelwidth{\@biblabel{#1}}%
%             \leftmargin\labelwidth
%             \advance\leftmargin\labelsep
%             \@openbib@code
%             \usecounter{enumiv}%
%             \let\p@enumiv\@empty
%             \renewcommand\theenumiv{\@arabic\c@enumiv}
%             \addcontentsline{toc}{chapter}{\bibname}}%
%       \sloppy\clubpenalty4000\widowpenalty4000%
%       \sfcode`\.\@m}
%      {\def\@noitemerr
%        {\@latex@warning{Empty `thebibliography' environment}}%
%       \endlist}

\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {{%
        \ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter: \ %
        \fi
        ##1}}{}}%
    \def\sectionmark##1{%
      \markright {{%
        \ifnum \c@secnumdepth >\z@
          \thesection: \ %
        \fi
        ##1}}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter: \ %
        \fi
        ##1}}}}
\fi


\if@twoside
\def\ps@iss{\let\@mkboth\gobbletwo \let\@oddhead\@empty \let\@evenhead\@empty
     \let\@oddfoot\@empty\def\@oddhead{\hfill \thepage
     \hfil}\def\@evenhead{\thepage \hfill}\let\@evenfoot\@oddfoot}
\else
\def\ps@iss{\let\@mkboth\gobbletwo \let\@oddhead\@empty \let\@evenhead\@empty
     \let\@oddfoot\@empty\def\@oddhead{\hfill \thepage
     \hfil}\def\@evenhead{\thepage}\let\@evenfoot\@oddfoot}
\fi

\renewcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{iss}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {%
    \parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \huge\bfseries \thechapter
    \fi
    \interlinepenalty\@M
    \huge \bfseries \quad #1\par\nobreak
    \vskip 40\p@
  }}

\renewcommand{\labelenumii}{\roman{enumii}.)}

\renewcommand{\arraystretch}{1.15}

\setcounter{tocdepth}{3}    %Alles bis subsubsection wird ins Inhaltsverzeichnis aufgenommen
\setcounter{secnumdepth}{3} %Alles bis subsubsection wird nummeriert

\setlength{\textwidth}{15.5cm}
\setlength{\textheight}{24.5cm}

\if@twoside
  \setlength{\oddsidemargin}{0.25cm}%
  \setlength{\evensidemargin}{-0.25cm}%
  \setlength{\marginparwidth}{80pt}%
\else
  \setlength{\oddsidemargin}{36pt}%
  \setlength{\evensidemargin}{36pt}%
  \setlength{\marginparwidth}{54pt}%
\fi

\setlength{\marginparsep}{10pt}
\setlength{\topmargin}{9pt}
\setlength{\headheight}{12pt}
\setlength{\headsep}{15pt}
\setlength{\footskip}{30pt}

\setlength{\columnsep}{10pt}
\setlength{\columnseprule}{0pt}

\frenchspacing

\setlength{\parindent}{0mm}
\setlength{\parskip}{0mm}

\addtolength{\topmargin}{-40.8pt} % verringert den oberen Rand

\sloppy

