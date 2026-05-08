use super::operator::Operator;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Number(f64),
    Column(String),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        op: Operator,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    End,
}

pub(crate) fn parse_expression(raw: &str) -> Result<Expr, String> {
    let tokens = tokenize(raw)?;
    let mut parser = Parser {
        tokens,
        position: 0,
    };
    let expr = parser.parse_expr()?;
    if parser.peek() != &Token::End {
        return Err(format!(
            "unexpected token after expression: {:?}",
            parser.peek()
        ));
    }
    Ok(expr)
}

fn tokenize(raw: &str) -> Result<Vec<Token>, String> {
    let chars = raw.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut idx = 0usize;
    while idx < chars.len() {
        let ch = chars[idx];
        if ch.is_whitespace() {
            idx += 1;
            continue;
        }
        match ch {
            '+' => {
                tokens.push(Token::Plus);
                idx += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                idx += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                idx += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                idx += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                idx += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                idx += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                idx += 1;
            }
            '0'..='9' | '.' => {
                let start = idx;
                idx += 1;
                while idx < chars.len()
                    && (chars[idx].is_ascii_digit()
                        || chars[idx] == '.'
                        || chars[idx] == 'e'
                        || chars[idx] == 'E'
                        || ((chars[idx] == '+' || chars[idx] == '-')
                            && matches!(chars[idx - 1], 'e' | 'E')))
                {
                    idx += 1;
                }
                let text = chars[start..idx].iter().collect::<String>();
                let value = text
                    .parse::<f64>()
                    .map_err(|err| format!("invalid number literal {text:?}: {err}"))?;
                tokens.push(Token::Number(value));
            }
            '_' | 'A'..='Z' | 'a'..='z' => {
                let start = idx;
                idx += 1;
                while idx < chars.len() && (chars[idx] == '_' || chars[idx].is_ascii_alphanumeric())
                {
                    idx += 1;
                }
                tokens.push(Token::Ident(chars[start..idx].iter().collect()));
            }
            _ => return Err(format!("unexpected character in expression: {ch:?}")),
        }
    }
    tokens.push(Token::End);
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::End)
    }

    fn bump(&mut self) -> Token {
        let token = self.peek().clone();
        self.position += 1;
        token
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_mul_div()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.bump();
            let right = self.parse_mul_div()?;
            expr = Expr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                _ => break,
            };
            self.bump();
            let right = self.parse_unary()?;
            expr = Expr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Token::Plus => {
                self.bump();
                self.parse_unary()
            }
            Token::Minus => {
                self.bump();
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.bump() {
            Token::Number(value) => Ok(Expr::Number(value)),
            Token::Ident(name) => {
                if self.peek() == &Token::LParen {
                    self.bump();
                    let mut args = Vec::new();
                    if self.peek() != &Token::RParen {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.peek() == &Token::Comma {
                                self.bump();
                                continue;
                            }
                            break;
                        }
                    }
                    if self.peek() != &Token::RParen {
                        return Err(format!("function {name} call missing closing ')'"));
                    }
                    self.bump();
                    Ok(Expr::Call {
                        op: Operator::from_name(&name)?,
                        args,
                    })
                } else {
                    Ok(Expr::Column(name))
                }
            }
            Token::LParen => {
                let expr = self.parse_expr()?;
                if self.peek() != &Token::RParen {
                    return Err("missing closing ')'".to_owned());
                }
                self.bump();
                Ok(expr)
            }
            token => Err(format!("unexpected token in expression: {token:?}")),
        }
    }
}
