"""Directorio de contactos ficticios para la demostración.

Estos contactos no pertenecen a una empresa real y no provienen
del corpus documental.
"""

from __future__ import annotations

from bimbam_assistant.domain.models import SupportContact


DEMO_SUPPORT_CONTACTS = {
    "general": SupportContact(
        area="Atención al cliente",
        email="soporte-bimbam@example.com",
    ),
    "envios": SupportContact(
        area="Soporte logístico",
        email="logistica-bimbam@example.com",
    ),
    "garantias": SupportContact(
        area="Garantías",
        email="garantias-bimbam@example.com",
    ),
    "reembolsos_devoluciones": SupportContact(
        area="Postventa y devoluciones",
        email="postventa-bimbam@example.com",
    ),
    "metodos_pago": SupportContact(
        area="Soporte de pagos",
        email="pagos-bimbam@example.com",
    ),
    "afiliados": SupportContact(
        area="Programa de afiliados",
        email="afiliados-bimbam@example.com",
    ),
}


def get_demo_support_contact(
    category: str | None = None,
) -> SupportContact:
    """Obtiene un contacto ficticio según la categoría."""

    contact = DEMO_SUPPORT_CONTACTS.get(
        category or "general",
        DEMO_SUPPORT_CONTACTS["general"],
    )

    return contact.model_copy(
        deep=True
    )