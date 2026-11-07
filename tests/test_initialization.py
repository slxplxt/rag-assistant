import ssl
from unittest.mock import Mock, patch

import pytest
import query_data


def test_llm_verifies_tls_by_default(monkeypatch):
    monkeypatch.delenv("GIGACHAT_VERIFY_SSL_CERTS", raising=False)
    with patch.object(query_data, "GIGACHAT_CREDENTIALS", "test-placeholder"):
        with patch.object(query_data, "GigaChat") as factory:
            query_data.init_llm()
    options = factory.call_args.kwargs
    assert options["verify_ssl_certs"] is True
    assert options["ssl_context"].verify_mode == ssl.CERT_REQUIRED
    assert options["ssl_context"].check_hostname is True


def test_explicit_local_tls_exception(monkeypatch):
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "false")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", "unused.pem")
    with patch.object(query_data, "GIGACHAT_CREDENTIALS", "test-placeholder"):
        with patch.object(query_data, "GigaChat") as factory:
            query_data.init_llm()
    options = factory.call_args.kwargs
    assert options["verify_ssl_certs"] is False
    assert options["ssl_context"] is None
    assert options["ca_bundle_file"] is None


def test_tls_typo_fails_closed(monkeypatch):
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "flase")
    with patch.object(query_data, "GIGACHAT_CREDENTIALS", "test-placeholder"):
        with pytest.raises(ValueError, match="true или false"):
            query_data.init_llm()


def test_trusted_ca_bundle_is_preserved(monkeypatch):
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "true")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", "trusted.pem")
    with patch.object(query_data, "GIGACHAT_CREDENTIALS", "test-placeholder"):
        with patch.object(query_data, "GigaChat") as factory:
            query_data.init_llm()
    assert factory.call_args.kwargs["ca_bundle_file"] == "trusted.pem"


@pytest.mark.parametrize("metadata", [{"bank": "vtb"}, {}, None])
def test_old_or_mixed_index_is_rejected(tmp_path, metadata):
    db = Mock()
    db.get.side_effect = [{"ids": ["id"]}, {"metadatas": [metadata]}]
    with patch.object(query_data, "DB_DIR", tmp_path):
        with patch.object(query_data, "HuggingFaceEmbeddings"):
            with patch.object(query_data, "Chroma", return_value=db):
                with pytest.raises(RuntimeError, match="индекс только Т-Банка"):
                    query_data.init_vector_db()
